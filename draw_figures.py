import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd

def draw_figures(epoch_list, train_acc_list, val_acc_list, loss_list, model_name):
    df = pd.DataFrame({
        'epochs': epoch_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'loss': loss_list
    })
    sns.set_theme(style='darkgrid')
    
    '''    
    # 使用系统已有的中文字体
    try:
        font = FontProperties(fname='SimHei')  # 尝试使用黑体
    except:
        font = FontProperties(fname='Microsoft YaHei')  # 如果黑体不可用，尝试使用微软雅黑
    '''
    # font_path = '/usr/share/fonts/truetype/arphic/SimHei.ttf'
    try:
        font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    except:
        font = FontProperties()
    
    # 创建2x2布局的子图
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 训练准确率
    sns.lineplot(x='epochs', y='train_acc', data=df, marker='o', ax=axs[0, 0])
    axs[0, 0].set_title('训练准确率曲线', fontproperties=font)
    axs[0, 0].set_xlabel('训练轮次', fontproperties=font)
    axs[0, 0].set_ylabel('准确率', fontproperties=font)
    axs[0, 0].grid(True)
    
    # 子图2: 验证准确率
    sns.lineplot(x='epochs', y='val_acc', data=df, marker='o', color='orange', ax=axs[0, 1])
    axs[0, 1].set_title('验证准确率曲线', fontproperties=font)
    axs[0, 1].set_xlabel('训练轮次', fontproperties=font)
    axs[0, 1].set_ylabel('准确率', fontproperties=font)
    axs[0, 1].grid(True)
    
    # 子图3: 损失曲线
    sns.lineplot(x='epochs', y='loss', data=df, marker='o', color='red', ax=axs[1, 0])
    axs[1, 0].set_title('训练损失曲线', fontproperties=font)
    axs[1, 0].set_xlabel('训练轮次', fontproperties=font)
    axs[1, 0].set_ylabel('损失值', fontproperties=font)
    axs[1, 0].grid(True)
    
    # 子图4: 训练和验证准确率对比
    sns.lineplot(x='epochs', y='train_acc', data=df, marker='o', label='训练准确率', ax=axs[1, 1])
    sns.lineplot(x='epochs', y='val_acc', data=df, marker='s', label='验证准确率', ax=axs[1, 1])
    axs[1, 1].set_title('训练与验证准确率对比', fontproperties=font)
    axs[1, 1].set_xlabel('训练轮次', fontproperties=font)
    axs[1, 1].set_ylabel('准确率', fontproperties=font)
    axs[1, 1].legend(prop=font)
    axs[1, 1].grid(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'./figures/{model_name}training_curves_summary.png', dpi=300)
    plt.close()