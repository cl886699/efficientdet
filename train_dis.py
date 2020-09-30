import numpy as np
import os
import tensorflow as tf
from nets.efficientdet_training import focal, smooth_l1
from nets.efficientdet import Efficientdet
from utils.utils import BBoxUtility, ModelCheckpoint
from utils.anchors import get_anchors
from tqdm import tqdm
import time
from getdata_tfrecord import ZipDataset


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

freeze_layers = [226, 328, 328, 373, 463, 463, 655, 802]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

class DistTrainer:
    def __init__(self, dis_strategy, ori_model, focal_loss, smooth_l1_loss, optimizer_in, trian_dir=''):
        self.dist_strategy = dis_strategy
        self.model = ori_model
        self.trian_dir = trian_dir
        self.optimizer = optimizer_in
        self.focal = focal_loss
        self.smooth_l1 = smooth_l1_loss

    @tf.function
    def train_step(self, imgs, targets0, targets1):
        with tf.GradientTape() as tape:
            # 计算loss
            regression, classification = self.model(imgs, training=True)
            reg_value = self.smooth_l1(targets0, regression)
            cls_value = self.focal(targets1, classification)
            loss_value = reg_value + cls_value
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value, cls_value, reg_value

    @tf.function
    def val_step(self, imgs, targets0, targets1):
        # 计算loss
        regression, classification = self.model(imgs)
        reg_value = self.smooth_l1 (targets0, regression)
        cls_value = self.focal(targets1, classification)
        loss_value = reg_value + cls_value
        return loss_value, cls_value, reg_value

    def dist_train_step(self, images, targets0, targets1):
        per_loss_value, per_rpn_class_loss, per_rpn_bbox_loss = self.dist_strategy.run(
            self.train_step,
            args=(images, targets0, targets1))
        loss_value = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_value, axis=None)
        rpn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_class_loss, axis=None)
        rpn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_bbox_loss, axis=None)
        return loss_value, rpn_class_loss, rpn_bbox_loss

    def dist_val_step(self, images, targets0, targets1):
        per_loss_value, per_rpn_class_loss, per_rpn_bbox_loss = self.dist_strategy.run(
            self.val_step,
            args=(images, targets0, targets1))
        loss_value = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_value, axis=None)
        rpn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_class_loss, axis=None)
        rpn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_bbox_loss, axis=None)
        return loss_value, rpn_class_loss, rpn_bbox_loss

    def fit_one_epoch(self, epoch, epoch_size, epoch_size_val, train_ds, val_ds, Epoch):
        total_r_loss = 0
        total_c_loss = 0
        total_loss = 0
        val_loss = 0
        start_time = time.time()
        gen = self.dist_strategy.experimental_distribute_dataset(train_ds)
        genval = self.dist_strategy.experimental_distribute_dataset(val_ds)
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets0, targets1 = batch[0], batch[1], batch[2]
                loss_value, reg_value, cls_value = self.dist_train_step(images, targets0, targets1)
                total_loss += loss_value
                total_c_loss += cls_value
                total_r_loss += reg_value
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'conf_loss': float(total_c_loss) / (iteration + 1),
                                    'regression_loss': float(total_r_loss) / (iteration + 1),
                                    'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()

        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                # 计算验证集loss
                images, targets0, targets1 = batch[0], batch[1], batch[2]
                loss_value, _, _ = self.dist_val_step(images, targets0, targets1)
                # 更新验证集loss
                val_loss = val_loss + loss_value
                pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
                pbar.update(1)

        print('Finish Validation')
        print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
        self.model.save_weights('logs3/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5' % (
            (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    PER_GPU_BATCHSIZE = 4
    dist_strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    num_devices = dist_strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(num_devices))
    GLOBAL_BATCHSIZE = num_devices * PER_GPU_BATCHSIZE

    with dist_strategy.scope():
        if os.name == 'nt':
            tf_record_path = 'D:/datasets/bjod/'
        else:
            tf_record_path = '../../../../datasets/bjod/'
        # -------------------------------------------#
        #   训练前，请指定好phi和model_path
        #   二者所使用Efficientdet版本要相同
        # -------------------------------------------#
        phi = 1
        classes_path = 'model_data/bjod.txt'
        class_names = get_classes(classes_path)
        NUM_CLASSES = len(class_names)
        # -------------------------------------------#
        #   权值文件的下载请看README
        # -------------------------------------------#
        model_path = "model_data/efficientdet-d1-voc.h5"
        # -------------------------------#
        #   Dataloder的使用
        # -------------------------------#
        model = Efficientdet(phi, num_classes=NUM_CLASSES)
        priors = get_anchors(image_sizes[phi])
        bbox_util = BBoxUtility(NUM_CLASSES, priors)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        # ------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   Init_Epoch为起始世代
        #   Freeze_Epoch为冻结训练的世代
        #   Epoch总训练世代
        # ------------------------------------------------------#
        for i in range(freeze_layers[phi]):
            model.layers[i].trainable = False

        if True:
            # --------------------------------------------#
            #   BATCH_SIZE不要太小，不然训练效果很差
            # --------------------------------------------#
            BATCH_SIZE = GLOBAL_BATCHSIZE
            Lr = 1e-3
            Init_Epoch = 0
            Freeze_Epoch = 60
            num_train = 97
            num_val = 42
            train_datasets, val_datasets = ZipDataset(tf_record_path, BATCH_SIZE, 96, bbox_util, 1, crop_size=[640, 640, 3],
                                        roi_path=os.path.join(tf_record_path, 'roi_test')).prepare(True)
            epoch_size = num_train // BATCH_SIZE
            epoch_size_val = num_val // (BATCH_SIZE//4)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=Lr,
                decay_steps=epoch_size,
                decay_rate=0.9,
                staircase=True
            )
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            trainer = DistTrainer(dist_strategy, model, focal_loss=focal(), smooth_l1_loss= smooth_l1(), optimizer_in=optimizer)
            for epoch in range(Init_Epoch, Freeze_Epoch):
                trainer.fit_one_epoch(epoch, epoch_size, epoch_size_val, train_datasets, val_datasets, Freeze_Epoch)

        for i in range(freeze_layers[phi]):
            model.layers[i].trainable = True

        if True:
            # --------------------------------------------#
            #   BATCH_SIZE不要太小，不然训练效果很差
            # --------------------------------------------#
            BATCH_SIZE = GLOBAL_BATCHSIZE//2
            Lr = 5e-5
            Freeze_Epoch = 50
            Epoch = 100
            train_datasets, val_datasets = ZipDataset(tf_record_path, BATCH_SIZE, 96, bbox_util, 1, crop_size=[640, 640, 3],
                                        roi_path=os.path.join(tf_record_path, 'roi_test')).prepare(True)
            epoch_size = num_train // BATCH_SIZE
            epoch_size_val = num_val // (BATCH_SIZE//4)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=Lr,
                decay_steps=epoch_size,
                decay_rate=0.9,
                staircase=True
            )
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            trainer = DistTrainer(dist_strategy, model, focal_loss=focal(), smooth_l1_loss= smooth_l1(), optimizer_in=optimizer)
            for epoch in range(Init_Epoch, Freeze_Epoch):
                trainer.fit_one_epoch(epoch, epoch_size, epoch_size_val, train_datasets, val_datasets, Freeze_Epoch)
