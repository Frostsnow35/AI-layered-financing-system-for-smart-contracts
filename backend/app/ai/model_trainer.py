"""
模型训练和预测模块
实现AI模型的训练、预测和评估功能
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    模型训练器类
    实现AI模型的训练、优化和评估
    """
    
    def __init__(self, model_save_path: str = "models/"):
        """
        初始化模型训练器
        
        Args:
            model_save_path: 模型保存路径
        """
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        
        # 模型
        self.rf_model = None
        self.gb_model = None
        self.stacking_model = None
        self.best_model = None
        
        # 训练历史
        self.training_history = {}
        
        # 确保模型保存目录存在
        os.makedirs(model_save_path, exist_ok=True)
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理
        
        Args:
            data: 原始数据DataFrame
            target_column: 目标列名
            
        Returns:
            处理后的特征矩阵和目标变量
        """
        logger.info("开始数据预处理...")
        
        # 分离特征和目标变量
        if target_column in data.columns:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
        else:
            X = data
            y = None
        
        # 处理缺失值
        X_imputed = self.imputer.fit_transform(X)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # 处理目标变量
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = None
        
        logger.info(f"数据预处理完成，特征维度: {X_scaled.shape}")
        
        return X_scaled, y_encoded
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, optimize: bool = True) -> Dict[str, float]:
        """
        训练随机森林模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            optimize: 是否进行超参数优化
            
        Returns:
            模型性能指标
        """
        logger.info("开始训练随机森林模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if optimize:
            # 超参数优化
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.rf_model = grid_search.best_estimator_
            logger.info(f"随机森林最佳参数: {grid_search.best_params_}")
        else:
            # 使用默认参数
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
        
        # 评估模型
        performance = self._evaluate_model(self.rf_model, X_test, y_test)
        performance['model_name'] = 'RandomForest'
        
        # 保存训练历史
        self.training_history['random_forest'] = {
            'performance': performance,
            'feature_importance': self.rf_model.feature_importances_
        }
        
        logger.info(f"随机森林训练完成，准确率: {performance['accuracy']:.3f}")
        
        return performance
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray, optimize: bool = True) -> Dict[str, float]:
        """
        训练梯度提升模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            optimize: 是否进行超参数优化
            
        Returns:
            模型性能指标
        """
        logger.info("开始训练梯度提升模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if optimize:
            # 超参数优化
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.gb_model = grid_search.best_estimator_
            logger.info(f"梯度提升最佳参数: {grid_search.best_params_}")
        else:
            # 使用默认参数
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.gb_model.fit(X_train, y_train)
        
        # 评估模型
        performance = self._evaluate_model(self.gb_model, X_test, y_test)
        performance['model_name'] = 'GradientBoosting'
        
        # 保存训练历史
        self.training_history['gradient_boosting'] = {
            'performance': performance,
            'feature_importance': self.gb_model.feature_importances_
        }
        
        logger.info(f"梯度提升训练完成，准确率: {performance['accuracy']:.3f}")
        
        return performance
    
    def train_stacking_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        训练Stacking融合模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            模型性能指标
        """
        logger.info("开始训练Stacking融合模型...")
        
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("请先训练随机森林和梯度提升模型")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 创建Stacking模型
        self.stacking_model = StackingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        # 训练模型
        self.stacking_model.fit(X_train, y_train)
        
        # 评估模型
        performance = self._evaluate_model(self.stacking_model, X_test, y_test)
        performance['model_name'] = 'Stacking'
        
        # 保存训练历史
        self.training_history['stacking'] = {
            'performance': performance
        }
        
        logger.info(f"Stacking模型训练完成，准确率: {performance['accuracy']:.3f}")
        
        return performance
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            性能指标字典
        """
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算性能指标
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return performance
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, optimize: bool = True) -> Dict[str, Dict[str, float]]:
        """
        训练所有模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            optimize: 是否进行超参数优化
            
        Returns:
            所有模型的性能指标
        """
        logger.info("开始训练所有模型...")
        
        all_performance = {}
        
        # 训练随机森林
        rf_performance = self.train_random_forest(X, y, optimize)
        all_performance['random_forest'] = rf_performance
        
        # 训练梯度提升
        gb_performance = self.train_gradient_boosting(X, y, optimize)
        all_performance['gradient_boosting'] = gb_performance
        
        # 训练Stacking模型
        stacking_performance = self.train_stacking_model(X, y)
        all_performance['stacking'] = stacking_performance
        
        # 选择最佳模型
        best_model_name = max(all_performance.keys(), 
                             key=lambda x: all_performance[x]['f1_score'])
        
        if best_model_name == 'random_forest':
            self.best_model = self.rf_model
        elif best_model_name == 'gradient_boosting':
            self.best_model = self.gb_model
        else:
            self.best_model = self.stacking_model
        
        logger.info(f"最佳模型: {best_model_name}, F1分数: {all_performance[best_model_name]['f1_score']:.3f}")
        
        return all_performance
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            X: 特征矩阵
            model_name: 模型名称，None表示使用最佳模型
            
        Returns:
            预测结果和预测概率
        """
        if model_name is None:
            model = self.best_model
        elif model_name == 'random_forest':
            model = self.rf_model
        elif model_name == 'gradient_boosting':
            model = self.gb_model
        elif model_name == 'stacking':
            model = self.stacking_model
        else:
            raise ValueError(f"未知模型名称: {model_name}")
        
        if model is None:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        return y_pred, y_pred_proba
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性DataFrame
        """
        if model_name is None:
            model_name = 'random_forest'  # 默认使用随机森林
        
        if model_name not in self.training_history:
            raise ValueError(f"模型 {model_name} 尚未训练")
        
        importance = self.training_history[model_name]['feature_importance']
        
        # 创建特征名称（这里简化处理）
        feature_names = [f"factor_{i+1}" for i in range(len(importance))]
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, model_name: str = 'random_forest', top_n: int = 20):
        """
        绘制特征重要性图
        
        Args:
            model_name: 模型名称
            top_n: 显示前N个重要特征
        """
        importance_df = self.get_feature_importance(model_name)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'{model_name} 特征重要性 (Top {top_n})')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self):
        """
        绘制模型性能对比图
        """
        if not self.training_history:
            logger.warning("没有训练历史数据")
            return
        
        # 提取性能指标
        models = list(self.training_history.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # 创建数据
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'model': model,
                    'metric': metric,
                    'score': self.training_history[model]['performance'][metric]
                })
        
        df = pd.DataFrame(data)
        
        # 绘制对比图
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='metric', y='score', hue='model')
        plt.title('模型性能对比')
        plt.xlabel('性能指标')
        plt.ylabel('分数')
        plt.legend(title='模型')
        plt.tight_layout()
        plt.show()
    
    def save_models(self):
        """保存训练好的模型"""
        if not self.training_history:
            raise ValueError("没有训练好的模型可以保存")
        
        logger.info("保存模型...")
        
        # 保存模型
        if self.rf_model is not None:
            joblib.dump(self.rf_model, os.path.join(self.model_save_path, 'rf_model.pkl'))
        
        if self.gb_model is not None:
            joblib.dump(self.gb_model, os.path.join(self.model_save_path, 'gb_model.pkl'))
        
        if self.stacking_model is not None:
            joblib.dump(self.stacking_model, os.path.join(self.model_save_path, 'stacking_model.pkl'))
        
        if self.best_model is not None:
            joblib.dump(self.best_model, os.path.join(self.model_save_path, 'best_model.pkl'))
        
        # 保存预处理器
        joblib.dump(self.scaler, os.path.join(self.model_save_path, 'scaler.pkl'))
        joblib.dump(self.imputer, os.path.join(self.model_save_path, 'imputer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.model_save_path, 'label_encoder.pkl'))
        
        # 保存训练历史
        joblib.dump(self.training_history, os.path.join(self.model_save_path, 'training_history.pkl'))
        
        logger.info("模型保存完成")
    
    def load_models(self):
        """加载已训练的模型"""
        try:
            logger.info("加载模型...")
            
            # 加载模型
            if os.path.exists(os.path.join(self.model_save_path, 'rf_model.pkl')):
                self.rf_model = joblib.load(os.path.join(self.model_save_path, 'rf_model.pkl'))
            
            if os.path.exists(os.path.join(self.model_save_path, 'gb_model.pkl')):
                self.gb_model = joblib.load(os.path.join(self.model_save_path, 'gb_model.pkl'))
            
            if os.path.exists(os.path.join(self.model_save_path, 'stacking_model.pkl')):
                self.stacking_model = joblib.load(os.path.join(self.model_save_path, 'stacking_model.pkl'))
            
            if os.path.exists(os.path.join(self.model_save_path, 'best_model.pkl')):
                self.best_model = joblib.load(os.path.join(self.model_save_path, 'best_model.pkl'))
            
            # 加载预处理器
            self.scaler = joblib.load(os.path.join(self.model_save_path, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_save_path, 'imputer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_save_path, 'label_encoder.pkl'))
            
            # 加载训练历史
            self.training_history = joblib.load(os.path.join(self.model_save_path, 'training_history.pkl'))
            
            logger.info("模型加载完成")
            
        except FileNotFoundError as e:
            logger.warning(f"模型文件不存在: {str(e)}")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    print("模型训练模块加载成功")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 110
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 测试模型训练
    trainer = ModelTrainer()
    
    # 训练所有模型
    performance = trainer.train_all_models(X, y, optimize=False)
    
    print("模型训练测试完成")
    for model_name, perf in performance.items():
        print(f"{model_name}: 准确率={perf['accuracy']:.3f}, F1分数={perf['f1_score']:.3f}")
    
    # 测试预测
    test_X = np.random.rand(10, n_features)
    y_pred, y_pred_proba = trainer.predict(test_X)
    
    print(f"预测结果: {y_pred}")
    print(f"预测概率: {y_pred_proba}")
    
    # 保存模型
    trainer.save_models()
    print("模型保存完成")
