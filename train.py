import os  # 导入os模块，用于处理文件和目录
import time  # 导入time模块，用于时间操作
from tensorboardX import SummaryWriter  # 从tensorboardX导入SummaryWriter，用于记录训练日志

from validate import validate  # 导入validate模块，用于模型验证
from data import create_dataloader  # 导入create_dataloader模块，用于创建数据加载器
from early_stopping_pytorch import EarlyStopping # 导入EarlyStopping模块，用于实现早停机制
from networks.trainer import Trainer  # 导入Trainer模块，用于训练模型
from options.train_options import TrainOptions  # 导入TrainOptions模块，用于解析训练选项

"""当前假设jpg_prob和blur_prob的值为0或1"""
def get_val_opt():  # 定义获取验证选项的函数
    val_opt = TrainOptions().parse(print_options=False)  # 解析验证选项，不打印选项
    val_opt.isTrain = False  # 设置为验证模式
    val_opt.no_resize = False  # 不进行图像缩放
    val_opt.no_crop = False  # 不进行裁剪
    val_opt.serial_batches = True  # 使用串行批处理
    val_opt.data_label = 'val'  # 设置数据标签为'val'
    val_opt.jpg_method = ['pil']  # 设置JPEG处理方法
    if len(val_opt.blur_sig) == 2:  # 如果模糊信号长度为2
        b_sig = val_opt.blur_sig  # 获取模糊信号
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]  # 取模糊信号的平均值
    if len(val_opt.jpg_qual) != 1:  # 如果JPEG质量不为1
        j_qual = val_opt.jpg_qual  # 获取JPEG质量
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]  # 取JPEG质量的平均值

    return val_opt  # 返回验证选项


if __name__ == '__main__':  # 如果是主程序
    opt = TrainOptions().parse()  # 解析训练选项
    val_opt = get_val_opt()  # 获取验证选项
 
    model = Trainer(opt)  # 创建Trainer实例

    data_loader = create_dataloader(opt)  # 创建训练数据加载器
    val_loader = create_dataloader(val_opt)  # 创建验证数据加载器

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))  # 创建训练日志记录器
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))  # 创建验证日志记录器
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)  # 初始化早停机制
    start_time = time.time()  # 记录开始时间
    print ("Length of data loader: %d" %(len(data_loader)))  # 打印数据加载器的长度
    for epoch in range(opt.niter):  # 遍历每个训练周期
        
        for i, data in enumerate(data_loader):  # 遍历训练数据
            model.total_steps += 1  # 增加总步数

            model.set_input(data)  # 设置模型输入
            model.optimize_parameters()  # 优化模型参数

            if model.total_steps % opt.loss_freq == 0:  # 如果达到损失记录频率
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))  # 打印训练损失
                train_writer.add_scalar('loss', model.loss, model.total_steps)  # 记录训练损失到TensorBoard
                print("Iter time: ", ((time.time()-start_time)/model.total_steps))  # 打印每次迭代的平均时间

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False:  # 在特定迭代步数下保存模型（当前条件为False，实际上不会执行）
                model.save_networks('model_iters_%s.pth' % model.total_steps)  # 保存模型

        if epoch % opt.save_epoch_freq == 0:  # 如果达到保存频率
            print('saving the model at the end of epoch %d' % (epoch))  # 打印保存模型的消息
            model.save_networks('model_epoch_best.pth')  # 保存最佳模型
            model.save_networks('model_epoch_%s.pth' % epoch)  # 保存当前周期模型

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)  # 在验证集上进行验证
        val_writer.add_scalar('accuracy', acc, model.total_steps)  # 记录准确率到TensorBoard
        val_writer.add_scalar('ap', ap, model.total_steps)  # 记录平均精度到TensorBoard
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))  # 打印验证结果

        early_stopping(acc, model)  # 检查早停条件
        if early_stopping.early_stop:  # 如果触发早停
            cont_train = model.adjust_learning_rate()  # 调整学习率
            if cont_train:  # 如果继续训练
                print("Learning rate dropped by 10, continue training...")  # 打印学习率下降的信息
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)  # 重新初始化早停机制
            else:  # 如果不继续训练
                print("Early stopping.")  # 打印早停信息
                break  # 退出循环
        model.train()  # 重新设置模型为训练模式
