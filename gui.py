import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from models import ConvNet3, ConvNet5, ResNet, ConvNetMix
from data_stats_cal import calculate_dataset_stats

class CalligraphyClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("楷书风格识别系统")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = None
        self.processed_tensor = None
        self.prediction = None
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['liu', 'ou', 'yan', 'zhao']  # 默认类别名称
        self.mean, self.std = None, None
        
        self.setup_ui()
        self.load_stats()
        
    def load_stats(self):
        """加载数据集统计信息用于图像标准化"""
        try:
            self.mean, self.std = calculate_dataset_stats("./dataset")
            print(f"已加载数据集统计信息: mean={self.mean}, std={self.std}")
        except Exception as e:
            print(f"加载数据集统计信息失败: {e}")
            # 使用默认值
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
            print(f"使用默认统计信息: mean={self.mean}, std={self.std}")
        
    def setup_ui(self):
        """设置用户界面"""
        # 顶部框架 - 模型选择
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(top_frame, text="选择模型:", bg="#f0f0f0", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        model_options = ["ConvNet3", "ConvNet5", "ConvNetMix", "ResNet"]
        self.model_var = tk.StringVar(value=model_options[0])
        model_dropdown = ttk.Combobox(top_frame, textvariable=self.model_var, values=model_options, width=15, state="readonly")
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        load_model_btn = tk.Button(top_frame, text="加载模型", command=self.load_model, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=10)
        load_model_btn.pack(side=tk.LEFT, padx=20)
        
        # 中间框架 - 图片显示
        middle_frame = tk.Frame(self.root, bg="#f0f0f0")
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧 - 图片显示
        left_frame = tk.Frame(middle_frame, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="图片预览", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=5)
        self.image_label = tk.Label(left_frame, bg="white", bd=1, relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        select_img_btn = tk.Button(left_frame, text="选择图片", command=self.select_image, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), padx=10)
        select_img_btn.pack(pady=10)
        
        # 右侧 - 结果显示
        right_frame = tk.Frame(middle_frame, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="识别结果", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.result_text = tk.Text(right_frame, height=10, width=30, font=("Arial", 12))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 真实标签框架
        label_frame = tk.Frame(right_frame, bg="#f0f0f0")
        label_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(label_frame, text="真实标签:", bg="#f0f0f0", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        
        self.true_label_var = tk.StringVar()
        label_dropdown = ttk.Combobox(label_frame, textvariable=self.true_label_var, values=self.class_names, width=10, state="readonly")
        label_dropdown.pack(side=tk.LEFT, padx=5)
        
        verify_btn = tk.Button(right_frame, text="验证预测", command=self.verify_prediction, bg="#FF9800", fg="white", font=("Arial", 10, "bold"), padx=10)
        verify_btn.pack(pady=10)
        
        # 状态栏
        status_frame = tk.Frame(self.root, bg="#e0e0e0", height=25)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="准备就绪", bg="#e0e0e0", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 初始化变量
        self.image_path = None
        self.processed_tensor = None
        self.prediction = None
        
    def load_model(self):
        """加载选择的模型"""
        model_name = self.model_var.get()
        try:
            # 加载类别名称（假设从数据集目录结构中获取）
            dataset_dir = "./dataset"
            if os.path.exists(dataset_dir):
                self.class_names = [d for d in os.listdir(dataset_dir) 
                                   if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
                print(f"加载类别名称: {self.class_names}")
            
            # 创建模型实例
            num_classes = len(self.class_names)
            if model_name == "ConvNet3":
                self.model = ConvNet3(num_classes=num_classes)
            elif model_name == "ConvNet5":
                self.model = ConvNet5(num_classes=num_classes)
            elif model_name == "ResNet":
                self.model = ResNet(num_classes=num_classes, input_channels=3)
            elif model_name == "ConvNetMix":
                self.model = ConvNetMix(num_classes=num_classes)
            
            # 加载模型权重
            model_path = f"./ckpt/{model_name}_final_model.pth"
            if not os.path.exists(model_path):
                model_path = f"./ckpt/{model_name}_best_model.pth"
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # 更新状态
            self.status_label.config(text=f"模型 {model_name} 加载成功")
            messagebox.showinfo("成功", f"模型 {model_name} 加载成功!")
            
        except Exception as e:
            self.status_label.config(text=f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            print(f"模型加载错误: {e}")
    
    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")]
        )
        
        if file_path:
            self.image_path = file_path
            # 显示原始图像
            self.display_image(file_path)
            # 处理图像并进行预测
            self.process_and_predict()
    
    def display_image(self, image_path):
        """显示图像"""
        try:
            # 打开并调整图像大小以适应显示区域
            img = Image.open(image_path)
            img.thumbnail((300, 300))  # 保持宽高比例缩放
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图像: {str(e)}")
    
    def process_and_predict(self):
        """处理图像并进行预测"""
        if not self.model:
            messagebox.showwarning("警告", "请先加载模型!")
            return
            
        if not self.image_path:
            return
            
        try:
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # 假设模型输入大小为64x64
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            
            # 打开并预处理图像
            img = Image.open(self.image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
            self.processed_tensor = img_tensor
            
            # 进行预测
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                self.prediction = probabilities
                
                # 获取预测类别和概率
                pred_class_idx = torch.argmax(probabilities).item()
                pred_class = self.class_names[pred_class_idx]
                pred_prob = probabilities[pred_class_idx].item()
                
                # 显示结果
                self.show_results(pred_class, pred_prob, probabilities)
        
        except Exception as e:
            self.status_label.config(text=f"预测失败: {str(e)}")
            messagebox.showerror("错误", f"预测过程中出错: {str(e)}")
            print(f"预测错误: {e}")
    
    def show_results(self, pred_class, pred_prob, probabilities):
        """显示预测结果"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"预测类别: {pred_class}\n")
        self.result_text.insert(tk.END, f"预测置信度: {pred_prob*100:.2f}%\n\n")
        self.result_text.insert(tk.END, "各类别概率:\n")
        
        for i, prob in enumerate(probabilities):
            if i < len(self.class_names):
                class_name = self.class_names[i]
                self.result_text.insert(tk.END, f"{class_name}: {prob.item()*100:.2f}%\n")
        
        self.status_label.config(text=f"已识别图像，预测类别: {pred_class}")
    
    def verify_prediction(self):
        """验证预测结果是否与真实标签匹配"""
        # 检查是否已进行过预测（张量不能直接进行布尔判断）
        if self.prediction is None:
            messagebox.showwarning("警告", "请先进行预测!")
            return
            
        true_label = self.true_label_var.get()
        if not true_label:
            messagebox.showwarning("警告", "请选择真实标签!")
            return
            
        # 获取预测类别
        pred_class_idx = torch.argmax(self.prediction).item()
        pred_class = self.class_names[pred_class_idx]
        
        # 判断预测是否正确
        is_correct = (pred_class == true_label)
        
        # 在结果文本中添加验证结果
        self.result_text.insert(tk.END, "\n验证结果:\n")
        self.result_text.insert(tk.END, f"真实标签: {true_label}\n")
        
        if is_correct:
            self.result_text.insert(tk.END, "预测结果: 正确 ✓")
            self.status_label.config(text=f"预测正确! 预测:{pred_class}, 实际:{true_label}")
        else:
            self.result_text.insert(tk.END, "预测结果: 错误 ✗")
            self.status_label.config(text=f"预测错误! 预测:{pred_class}, 实际:{true_label}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalligraphyClassifierGUI(root)
    root.mainloop()