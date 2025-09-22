"""
AI风险评分模型 - 核心模块2
实现养老服务分层融资的AI风险评分和智能匹配
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from app.core.config import settings

logger = logging.getLogger(__name__)


class RiskPredictionModel:
    """风险预测模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # 模型配置 - 专注于逻辑回归和随机森林
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'weight': 0.4
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'weight': 0.6
            }
        }
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征工程和数据预处理"""
        try:
            df = data.copy()
            
            # 处理缺失值
            df = self._handle_missing_values(df)
            
            # 特征编码
            df = self._encode_categorical_features(df)
            
            # 特征构造
            df = self._create_engineered_features(df)
            
            # 特征选择
            df = self._select_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"特征工程失败: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 数值型特征用中位数填充
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # 分类型特征用众数填充
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类型特征"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构造新特征"""
        # 服务收入稳定性指标
        if 'monthly_revenue' in df.columns:
            df['revenue_stability'] = df['monthly_revenue'].rolling(window=6).std()
            df['revenue_growth_rate'] = df['monthly_revenue'].pct_change()
        
        # 应收账款周转率
        if 'accounts_receivable' in df.columns and 'monthly_revenue' in df.columns:
            df['receivables_turnover'] = df['monthly_revenue'] / (df['accounts_receivable'] + 1)
        
        # 资产负债率
        if 'total_assets' in df.columns and 'total_liabilities' in df.columns:
            df['debt_to_asset_ratio'] = df['total_liabilities'] / (df['total_assets'] + 1)
        
        # 运营效率指标
        if 'operating_expenses' in df.columns and 'monthly_revenue' in df.columns:
            df['operating_efficiency'] = df['monthly_revenue'] / (df['operating_expenses'] + 1)
        
        # 客户满意度加权评分
        if 'service_quality_score' in df.columns and 'customer_satisfaction' in df.columns:
            df['weighted_satisfaction'] = (
                df['service_quality_score'] * 0.6 + 
                df['customer_satisfaction'] * 0.4
            )
        
        # 风险等级分类
        if 'risk_score' in df.columns:
            df['risk_level'] = pd.cut(
                df['risk_score'], 
                bins=[0, 40, 60, 80, 100], 
                labels=['高风险', '中高风险', '中低风险', '低风险']
            )
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征选择"""
        # 定义核心特征 - 专注于养老服务融资场景
        core_features = [
            'service_amount',           # 服务金额
            'service_duration',         # 服务时长
            'monthly_revenue',          # 月收入
            'service_quality_score',   # 服务质量评分
            'government_subsidy_rate', # 政府补贴率
            'revenue_stability',       # 收入稳定性
            'receivables_turnover',    # 应收账款周转率
            'debt_to_asset_ratio'      # 资产负债率
        ]
        
        # 选择存在的特征
        available_features = [col for col in core_features if col in df.columns]
        
        return df[available_features]
    
    def train_traditional_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练传统机器学习模型"""
        try:
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            performance_scores = {}
            
            # 训练各个模型
            for name, config in self.model_configs.items():
                model = config['model']
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 评估性能
                auc_score = roc_auc_score(y_test, y_pred_proba)
                performance_scores[name] = auc_score
                
                # 保存模型
                self.models[name] = model
                
                # 特征重要性（适用于树模型）
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                logger.info(f"{name} 模型训练完成，AUC: {auc_score:.4f}")
            
            self.model_performance = performance_scores
            return performance_scores
            
        except Exception as e:
            logger.error(f"传统模型训练失败: {str(e)}")
            raise
    
    def train_fund_matching_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """训练基金匹配模型"""
        try:
            # 数据标准化
            X_scaled = self.scaler.transform(X)
            
            # 分割训练和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 使用随机森林进行匹配预测
            matching_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            
            matching_model.fit(X_train, y_train)
            
            # 评估性能
            y_pred_proba = matching_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 保存匹配模型
            self.models['fund_matching'] = matching_model
            
            logger.info(f"基金匹配模型训练完成，AUC: {auc_score:.4f}")
            
            return auc_score
            
        except Exception as e:
            logger.error(f"基金匹配模型训练失败: {str(e)}")
            raise
    
    def predict_risk(self, features: Dict[str, float]) -> Dict[str, float]:
        """预测风险"""
        try:
            # 转换为DataFrame
            df = pd.DataFrame([features])
            
            # 特征工程
            df = self.prepare_features(df)
            
            # 转换为numpy数组
            X = df.values
            
            # 数据标准化
            X_scaled = self.scaler.transform(X)
            
            predictions = {}
            
            # 传统模型预测
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[0, 1]
                    predictions[name] = prob
            
            # 神经网络预测
            if self.nn_model:
                nn_prob = self.nn_model.predict(X_scaled)[0, 0]
                predictions['neural_network'] = nn_prob
            
            # 加权平均
            weighted_score = 0
            total_weight = 0
            
            for name, prob in predictions.items():
                if name in self.model_configs:
                    weight = self.model_configs[name]['weight']
                elif name == 'neural_network':
                    weight = 0.3
                else:
                    weight = 0.1
                
                weighted_score += prob * weight
                total_weight += weight
            
            final_score = weighted_score / total_weight if total_weight > 0 else 0
            
            # 风险等级判断
            risk_level = self._determine_risk_level(final_score)
            
            return {
                'risk_score': final_score,
                'risk_level': risk_level,
                'individual_predictions': predictions,
                'is_approved': final_score >= settings.AI_MODEL_THRESHOLD
            }
            
        except Exception as e:
            logger.error(f"风险预测失败: {str(e)}")
            raise
    
    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        if score >= 0.8:
            return "低风险"
        elif score >= 0.6:
            return "中低风险"
        elif score >= 0.4:
            return "中高风险"
        else:
            return "高风险"
    
    def get_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """获取特征重要性"""
        importance_dict = {}
        
        for model_name, importance in self.feature_importance.items():
            # 获取特征名称（这里简化处理）
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            # 排序
            sorted_importance = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            importance_dict[model_name] = sorted_importance
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            model_data = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'models': self.models,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'model_configs': self.model_configs
            }
            
            joblib.dump(model_data, filepath)
            
            # 保存神经网络模型
            if self.nn_model:
                nn_filepath = filepath.replace('.pkl', '_nn.h5')
                self.nn_model.save(nn_filepath)
            
            logger.info(f"模型保存成功: {filepath}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.models = model_data['models']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            self.model_configs = model_data['model_configs']
            
            # 加载神经网络模型
            nn_filepath = filepath.replace('.pkl', '_nn.h5')
            try:
                self.nn_model = keras.models.load_model(nn_filepath)
            except:
                logger.warning("神经网络模型文件不存在")
            
            logger.info(f"模型加载成功: {filepath}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        try:
            X_scaled = self.scaler.transform(X)
            
            evaluation_results = {}
            
            # 评估传统模型
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                    auc_score = roc_auc_score(y, y_pred_proba)
                    evaluation_results[name] = auc_score
            
            # 评估神经网络
            if self.nn_model:
                y_pred_proba = self.nn_model.predict(X_scaled).flatten()
                auc_score = roc_auc_score(y, y_pred_proba)
                evaluation_results['neural_network'] = auc_score
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise


# 创建全局模型实例
risk_model = RiskPredictionModel()
