"""
AI决策引擎核心模块
实现养老服务提供商信用AI智能评分模型的决策引擎
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIDecisionEngine:
    """
    AI决策引擎类
    实现基于七大维度110个影响因子的信用评分模型
    """
    
    def __init__(self, model_save_path: str = "models/"):
        """
        初始化AI决策引擎
        
        Args:
            model_save_path: 模型保存路径
        """
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # 七大维度权重
        self.dimension_weights = {
            'compliance': 0.25,      # 合规经营维度
            'service_quality': 0.30, # 服务质量维度
            'performance': 0.20,     # 履约能力维度
            'social_evaluation': 0.10, # 社会评价维度
            'risk_resistance': 0.08,   # 风险抵御维度
            'innovation': 0.05,         # 创新与可持续维度
            'special_scenario': 0.02   # 特殊场景补充因子
        }
        
        # 各维度内部因子权重
        self.factor_weights = self._init_factor_weights()
        
        # 模型
        self.rf_model = None
        self.gb_model = None
        self.stacking_model = None
        self.is_trained = False
        
        # 确保模型保存目录存在
        os.makedirs(model_save_path, exist_ok=True)
    
    def _init_factor_weights(self) -> Dict[str, Dict[str, float]]:
        """
        初始化各维度内部因子权重
        
        Returns:
            因子权重字典
        """
        return {
            'compliance': {
                # 资质合规类因子（70%）
                'license_validity': 0.20,
                'business_scope': 0.15,
                'fire_safety': 0.25,
                'food_license': 0.10,
                'medical_license': 0.10,
                'nurse_certification': 0.10,
                'medical_registration': 0.05,
                'background_check': 0.05,
                # 监管记录类因子（30%）
                'penalty_count': 0.30,
                'warning_count': 0.20,
                'inspection_pass_rate': 0.20,
                'rectification_rate': 0.15,
                'lawsuit_rate': 0.10,
                'lawsuit_loss_rate': 0.03,
                'contract_performance': 0.02
            },
            'service_quality': {
                # 安全保障类因子（35%）
                'accident_rate': 0.40,
                'accident_satisfaction': 0.25,
                'emergency_response': 0.20,
                'medical_error_rate': 0.10,
                'nursing_accident_rate': 0.05,
                # 服务执行类因子（25%）
                'care_plan_execution': 0.30,
                'health_monitoring': 0.25,
                'service_standard': 0.25,
                'service_completion': 0.15,
                'medical_accessibility': 0.05,
                # 投诉处理类因子（20%）
                'complaint_resolution': 0.40,
                'complaint_repeat': 0.30,
                'complaint_response_time': 0.20,
                'complaint_satisfaction': 0.10,
                # 满意度评价类因子（20%）
                'elder_satisfaction': 0.40,
                'family_satisfaction': 0.30,
                'long_term_stay_rate': 0.15,
                'transfer_rate': 0.10,
                'service_quality_score': 0.05
            },
            'performance': {
                # 财务健康类因子（50%）
                'revenue_growth': 0.25,
                'debt_ratio': 0.25,
                'fund_proportion': 0.20,
                'prepayment_compliance': 0.15,
                'social_security_rate': 0.10,
                'cash_flow': 0.03,
                'asset_turnover': 0.02,
                # 资源配置类因子（30%）
                'bed_utilization': 0.30,
                'nurse_ratio': 0.30,
                'medical_staff_ratio': 0.20,
                'equipment_condition': 0.15,
                'it_level': 0.03,
                'partner_stability': 0.02,
                # 运营稳定性类因子（20%）
                'management_stability': 0.30,
                'staff_turnover': 0.25,
                'service_interruption': 0.20,
                'service_recovery': 0.15,
                'strategic_planning': 0.10
            }
            # 其他维度的权重可以继续添加...
        }
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            处理后的特征矩阵和目标变量
        """
        logger.info("开始数据预处理...")
        
        # 分离特征和目标变量
        if 'default' in data.columns:
            X = data.drop('default', axis=1)
            y = data['default']
        else:
            X = data
            y = None
        
        # 处理缺失值
        X_imputed = self.imputer.fit_transform(X)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        logger.info(f"数据预处理完成，特征维度: {X_scaled.shape}")
        
        return X_scaled, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            模型性能指标
        """
        logger.info("开始训练模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练随机森林模型
        logger.info("训练随机森林模型...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # 训练梯度提升模型
        logger.info("训练梯度提升模型...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        
        # 训练Stacking模型
        logger.info("训练Stacking融合模型...")
        self.stacking_model = StackingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        self.stacking_model.fit(X_train, y_train)
        
        # 评估模型性能
        models = {
            'RandomForest': self.rf_model,
            'GradientBoosting': self.gb_model,
            'Stacking': self.stacking_model
        }
        
        performance = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info(f"{name} 性能: 准确率={performance[name]['accuracy']:.3f}, "
                       f"F1分数={performance[name]['f1_score']:.3f}, "
                       f"AUC={performance[name]['auc']:.3f}")
        
        self.is_trained = True
        
        # 保存模型
        self.save_models()
        
        return performance
    
    def predict_credit_score(self, features: np.ndarray) -> Dict[str, float]:
        """
        预测信用评分
        
        Args:
            features: 特征向量（110维）
            
        Returns:
            信用评分结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_models方法")
        
        # 确保特征维度正确
        if features.shape[1] != 110:
            raise ValueError(f"特征维度应为110，实际为{features.shape[1]}")
        
        # 数据预处理
        try:
            features_imputed = self.imputer.transform(features)
            features_scaled = self.scaler.transform(features_imputed)
        except:
            # 如果预处理器未拟合，使用原始特征
            features_scaled = features
        
        # 预测概率
        rf_proba = self.rf_model.predict_proba(features_scaled)[:, 1]
        gb_proba = self.gb_model.predict_proba(features_scaled)[:, 1]
        stacking_proba = self.stacking_model.predict_proba(features_scaled)[:, 1]
        
        # 加权融合
        final_proba = 0.6 * stacking_proba + 0.3 * rf_proba + 0.1 * gb_proba
        
        # 转换为0-100分
        credit_score = final_proba[0] * 100
        
        # 确定信用等级
        credit_level = self._get_credit_level(credit_score)
        
        return {
            'credit_score': credit_score,
            'credit_level': credit_level,
            'default_probability': final_proba[0],
            'rf_score': rf_proba[0] * 100,
            'gb_score': gb_proba[0] * 100,
            'stacking_score': stacking_proba[0] * 100
        }
    
    def _get_credit_level(self, score: float) -> str:
        """
        根据评分确定信用等级
        
        Args:
            score: 信用评分
            
        Returns:
            信用等级
        """
        if score >= 85:
            return "A级（优秀信用，风险极低）"
        elif score >= 70:
            return "B级（良好信用，风险较低）"
        elif score >= 25:
            return "C级（一般信用，风险中等）"
        elif score >= 10:
            return "D级（较差信用，风险较高）"
        else:
            return "E级（差信用，风险很高）"
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_models方法")
        
        # 获取特征重要性
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # 创建特征名称（这里简化处理，实际应该有110个特征名）
        feature_names = [f"factor_{i+1}" for i in range(110)]
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'rf_importance': rf_importance,
            'gb_importance': gb_importance
        })
        
        # 计算综合重要性
        importance_df['combined_importance'] = (
            0.5 * importance_df['rf_importance'] + 
            0.5 * importance_df['gb_importance']
        )
        
        # 按综合重要性排序
        importance_df = importance_df.sort_values('combined_importance', ascending=False)
        
        return importance_df
    
    def save_models(self):
        """保存训练好的模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        logger.info("保存模型...")
        
        # 保存模型
        joblib.dump(self.rf_model, os.path.join(self.model_save_path, 'rf_model.pkl'))
        joblib.dump(self.gb_model, os.path.join(self.model_save_path, 'gb_model.pkl'))
        joblib.dump(self.stacking_model, os.path.join(self.model_save_path, 'stacking_model.pkl'))
        
        # 保存预处理器
        joblib.dump(self.scaler, os.path.join(self.model_save_path, 'scaler.pkl'))
        joblib.dump(self.imputer, os.path.join(self.model_save_path, 'imputer.pkl'))
        
        logger.info("模型保存完成")
    
    def load_models(self):
        """加载已训练的模型"""
        try:
            logger.info("加载模型...")
            
            self.rf_model = joblib.load(os.path.join(self.model_save_path, 'rf_model.pkl'))
            self.gb_model = joblib.load(os.path.join(self.model_save_path, 'gb_model.pkl'))
            self.stacking_model = joblib.load(os.path.join(self.model_save_path, 'stacking_model.pkl'))
            
            self.scaler = joblib.load(os.path.join(self.model_save_path, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_save_path, 'imputer.pkl'))
            
            self.is_trained = True
            logger.info("模型加载完成")
            
        except FileNotFoundError:
            logger.warning("模型文件不存在，请先训练模型")
            self.is_trained = False


if __name__ == "__main__":
    # 测试代码
    print("AI决策引擎模块加载成功")
