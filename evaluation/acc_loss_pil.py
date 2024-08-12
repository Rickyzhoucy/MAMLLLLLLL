import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 接收两个文件夹路径
directory = "./m-1shot-finetuning"
directory_acc = directory + "/acc"  # 存放准确率数据的文件夹路径
directory_loss = directory + "/loss"  # 存放损失率数据的文件夹路径

# 查找所有准确率 CSV 文件
files_acc = os.listdir(directory_acc)
file_pre_train_acc = [f for f in files_acc if "train" in f and "Pre" in f][0]
file_post_train_acc = [f for f in files_acc if "train" in f and "Post" in f][0]
file_pre_eval_acc = [f for f in files_acc if "val" in f and "Pre" in f][0]
file_post_eval_acc = [f for f in files_acc if "val" in f and "Post" in f][0]

# 查找所有损失率 CSV 文件
files_loss = os.listdir(directory_loss)
file_pre_train_loss = [f for f in files_loss if "train" in f and "Pre" in f][0]
file_post_train_loss = [f for f in files_loss if "train" in f and "Post" in f][0]
file_pre_eval_loss = [f for f in files_loss if "val" in f and "Pre" in f][0]
file_post_eval_loss = [f for f in files_loss if "val" in f and "Post" in f][0]

# 读取 CSV 文件
data_pre_train_acc = pd.read_csv(directory_acc + '/' + file_pre_train_acc)
data_post_train_acc = pd.read_csv(directory_acc + '/' + file_post_train_acc)
data_pre_eval_acc = pd.read_csv(directory_acc + '/' + file_pre_eval_acc)
data_post_eval_acc = pd.read_csv(directory_acc + '/' + file_post_eval_acc)

data_pre_train_loss = pd.read_csv(directory_loss + '/' + file_pre_train_loss)
data_post_train_loss = pd.read_csv(directory_loss + '/' + file_post_train_loss)
data_pre_eval_loss = pd.read_csv(directory_loss + '/' + file_pre_eval_loss)
data_post_eval_loss = pd.read_csv(directory_loss + '/' + file_post_eval_loss)

# 提取迭代次数、准确率和损失率
iterations_pre_train = data_pre_train_acc['Step']
accuracies_pre_train = data_pre_train_acc['Value']
iterations_post_train = data_post_train_acc['Step']
accuracies_post_train = data_post_train_acc['Value']

iterations_pre_eval = data_pre_eval_acc['Step']
accuracies_pre_eval = data_pre_eval_acc['Value']
iterations_post_eval = data_post_eval_acc['Step']
accuracies_post_eval = data_post_eval_acc['Value']

loss_pre_train = data_pre_train_loss['Value']
loss_post_train = data_post_train_loss['Value']
loss_pre_eval = data_pre_eval_loss['Value']
loss_post_eval = data_post_eval_loss['Value']

# 创建画布和四个子图
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# 字体大小设置
title_fontsize = 16
label_fontsize = 14
ticks_fontsize = 12
legend_fontsize = 12

# 第一个子图：训练数据准确率
axs[0][0].plot(iterations_pre_train, accuracies_pre_train, label='Pre-training', color='blue')
axs[0][0].plot(iterations_post_train, accuracies_post_train, label='Post-training', color='red')
axs[0][0].set_title('Training Accuracy Over Iterations', fontsize=title_fontsize)
axs[0][0].set_xlabel('Iteration Number', fontsize=label_fontsize)
axs[0][0].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[0][0].set_yticks(np.arange(0, 1.05, 0.05))
axs[0][0].tick_params(axis='both', which='major', labelsize=ticks_fontsize)
axs[0][0].legend(fontsize=legend_fontsize)
axs[0][0].grid(True)

# 第二个子图：评估数据准确率
axs[1][0].plot(iterations_pre_eval, accuracies_pre_eval, label='Pre-evaluation', color='green')
axs[1][0].plot(iterations_post_eval, accuracies_post_eval, label='Post-evaluation', color='purple')
axs[1][0].set_title('Evaluation Accuracy Over Iterations', fontsize=title_fontsize)
axs[1][0].set_xlabel('Iteration Number', fontsize=label_fontsize)
axs[1][0].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[1][0].set_yticks(np.arange(0, 1.05, 0.05))
axs[1][0].tick_params(axis='both', which='major', labelsize=ticks_fontsize)
axs[1][0].legend(fontsize=legend_fontsize)
axs[1][0].grid(True)

# 第三个子图：训练数据损失率
axs[1][1].plot(iterations_pre_train, loss_pre_train, label='Pre-training Loss', color='blue', linestyle='--')
axs[1][1].plot(iterations_post_train, loss_post_train, label='Post-training Loss', color='red', linestyle='--')
axs[1][1].set_title('Training Loss Over Iterations', fontsize=title_fontsize)
axs[1][1].set_xlabel('Iteration Number', fontsize=label_fontsize)
axs[1][1].set_ylabel('Loss', fontsize=label_fontsize)
axs[1][1].set_yticks(np.arange(min(min(loss_pre_train), min(loss_post_train)), max(max(loss_pre_train), max(loss_post_train)) + 1, (max(max(loss_pre_train), max(loss_post_train)) - min(min(loss_pre_train), min(loss_post_train))) / 10))
axs[1][1].tick_params(axis='both', which='major', labelsize=ticks_fontsize)
axs[1][1].legend(fontsize=legend_fontsize)
axs[1][1].grid(True)

# 第四个子图：评估数据损失率
axs[0][1].plot(iterations_pre_eval, loss_pre_eval, label='Pre-evaluation Loss', color='green', linestyle='--')
axs[0][1].plot(iterations_post_eval, loss_post_eval, label='Post-evaluation Loss', color='purple', linestyle='--')
axs[0][1].set_title('Evaluation Loss Over Iterations', fontsize=title_fontsize)
axs[0][1].set_xlabel('Iteration Number', fontsize=label_fontsize)
axs[0][1].set_ylabel('Loss', fontsize=label_fontsize)
axs[0][1].set_yticks(np.arange(min(min(loss_pre_eval), min(loss_post_eval)), max(max(loss_pre_eval), max(loss_post_eval)) + 1, (max(max(loss_pre_eval), max(loss_post_eval)) - min(min(loss_pre_eval), min(loss_post_eval))) / 10))
axs[0][1].tick_params(axis='both', which='major', labelsize=ticks_fontsize)
axs[0][1].legend(fontsize=legend_fontsize)
axs[0][1].grid(True)

# 显示图表
plt.tight_layout()
plt.show()
