from .base_options import BaseOptions  # 从当前目录的base_options模块导入BaseOptions类

class TrainOptions(BaseOptions):  # 定义TrainOptions类，继承自BaseOptions
    def initialize(self, parser):  # 初始化方法，接受一个参数parser
        parser = BaseOptions.initialize(self, parser)  # 调用父类的initialize方法，传递parser
        parser.add_argument('--earlystop_epoch', type=int, default=5)  # 添加参数：早停的周期数，默认值为5
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')  # 添加参数：是否进行额外的数据增强（光度变化、模糊、JPEG压缩）
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')  # 添加参数：使用的优化器，默认值为'adam'
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')  # 添加参数：是否使用新的优化器，而不是加载优化器状态
        parser.add_argument('--loss_freq', type=int, default=10, help='frequency of showing loss on tensorboard')  # 添加参数：在TensorBoard上显示损失的频率，默认值为400
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')  # 添加参数：保存检查点的频率（周期数），默认值为1
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')  # 添加参数：起始周期计数，默认值为1
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler initialization')  # 添加参数：调度器初始化时的起始周期计数，默认值为-1
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')  # 添加参数：训练集的名称，默认值为'train'
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')  # 添加参数：验证集的名称，默认值为'val'
        parser.add_argument('--niter', type=int, default=100, help='total epoches')  # 添加参数：总训练周期数，默认值为100
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')  # 添加参数：Adam优化器的动量项，默认值为0.9
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')  # 添加参数：Adam优化器的初始学习率，默认值为0.0001

        self.isTrain = True  # 设置isTrain属性为True，表示这是训练模式
        return parser  # 返回修改后的parser
