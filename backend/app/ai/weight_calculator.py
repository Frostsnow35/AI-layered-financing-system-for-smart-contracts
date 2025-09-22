"""
权重计算模块
实现七大维度110个影响因子的权重计算
包括层次分析法(AHP)、熵权法、机器学习算法权重计算
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WeightCalculator:
    """
    权重计算器类
    实现多种权重计算方法
    """
    
    def __init__(self):
        """初始化权重计算器"""
        self.dimension_weights = {
            'compliance': 0.25,      # 合规经营维度
            'service_quality': 0.30, # 服务质量维度
            'performance': 0.20,     # 履约能力维度
            'social_evaluation': 0.10, # 社会评价维度
            'risk_resistance': 0.08,   # 风险抵御维度
            'innovation': 0.05,         # 创新与可持续维度
            'special_scenario': 0.02   # 特殊场景补充因子
        }
        
        # 权重计算方法系数
        self.method_weights = {
            'ahp': 0.3,      # 层次分析法
            'entropy': 0.2,  # 熵权法
            'ml': 0.3,       # 机器学习算法
            'expert': 0.2    # 专家打分法
        }
    
    def calculate_ahp_weights(self, comparison_matrix: np.ndarray) -> np.ndarray:
        """
        层次分析法(AHP)权重计算
        
        Args:
            comparison_matrix: 判断矩阵
            
        Returns:
            权重向量
        """
        logger.info("计算AHP权重...")
        
        n = comparison_matrix.shape[0]
        
        # 计算权重向量
        weights = np.zeros(n)
        for i in range(n):
            # 计算几何平均数
            weights[i] = np.power(np.prod(comparison_matrix[i, :]), 1/n)
        
        # 归一化
        weights = weights / np.sum(weights)
        
        # 一致性检验
        consistency_ratio = self._check_consistency(comparison_matrix, weights)
        logger.info(f"AHP权重计算完成，一致性比率: {consistency_ratio:.3f}")
        
        return weights
    
    def _check_consistency(self, matrix: np.ndarray, weights: np.ndarray) -> float:
        """
        一致性检验
        
        Args:
            matrix: 判断矩阵
            weights: 权重向量
            
        Returns:
            一致性比率
        """
        n = matrix.shape[0]
        
        # 计算最大特征值
        lambda_max = np.sum((matrix @ weights) / weights) / n
        
        # 计算一致性指标
        ci = (lambda_max - n) / (n - 1)
        
        # 随机一致性指标（RI值）
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
        ri = ri_values.get(n, 1.45)
        
        # 一致性比率
        cr = ci / ri
        
        return cr
    
    def calculate_entropy_weights(self, data: np.ndarray) -> np.ndarray:
        """
        熵权法权重计算
        
        Args:
            data: 数据矩阵 (样本数, 特征数)
            
        Returns:
            权重向量
        """
        logger.info("计算熵权法权重...")
        
        n_samples, n_features = data.shape
        
        # 数据标准化
        data_normalized = np.zeros_like(data)
        for j in range(n_features):
            min_val = np.min(data[:, j])
            max_val = np.max(data[:, j])
            if max_val != min_val:
                data_normalized[:, j] = (data[:, j] - min_val) / (max_val - min_val)
            else:
                data_normalized[:, j] = 1.0
        
        # 计算概率
        prob_matrix = np.zeros_like(data_normalized)
        for j in range(n_features):
            col_sum = np.sum(data_normalized[:, j])
            if col_sum > 0:
                prob_matrix[:, j] = data_normalized[:, j] / col_sum
            else:
                prob_matrix[:, j] = 1.0 / n_samples
        
        # 计算信息熵
        entropy = np.zeros(n_features)
        for j in range(n_features):
            for i in range(n_samples):
                if prob_matrix[i, j] > 0:
                    entropy[j] -= prob_matrix[i, j] * np.log(prob_matrix[i, j])
        
        # 计算权重
        weights = np.zeros(n_features)
        for j in range(n_features):
            weights[j] = (1 - entropy[j]) / np.sum(1 - entropy)
        
        logger.info("熵权法权重计算完成")
        return weights
    
    def calculate_ml_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        机器学习算法权重计算
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            各算法的特征重要性权重
        """
        logger.info("计算机器学习算法权重...")
        
        weights = {}
        
        # 随机森林特征重要性
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        weights['random_forest'] = rf_model.feature_importances_
        
        # 梯度提升特征重要性
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        gb_model.fit(X, y)
        weights['gradient_boosting'] = gb_model.feature_importances_
        
        # 综合权重
        weights['combined'] = 0.5 * weights['random_forest'] + 0.5 * weights['gradient_boosting']
        
        logger.info("机器学习算法权重计算完成")
        return weights
    
    def calculate_expert_weights(self, expert_scores: Dict[str, List[float]]) -> np.ndarray:
        """
        专家打分法权重计算
        
        Args:
            expert_scores: 专家打分字典 {因子名: [专家1分数, 专家2分数, ...]}
            
        Returns:
            权重向量
        """
        logger.info("计算专家打分法权重...")
        
        factor_names = list(expert_scores.keys())
        n_factors = len(factor_names)
        n_experts = len(expert_scores[factor_names[0]])
        
        # 计算每个因子的平均分
        avg_scores = np.zeros(n_factors)
        for i, factor in enumerate(factor_names):
            avg_scores[i] = np.mean(expert_scores[factor])
        
        # 归一化为权重
        weights = avg_scores / np.sum(avg_scores)
        
        logger.info("专家打分法权重计算完成")
        return weights
    
    def calculate_comprehensive_weights(self, 
                                      data: np.ndarray,
                                      y: Optional[np.ndarray] = None,
                                      expert_scores: Optional[Dict[str, List[float]]] = None) -> Dict[str, np.ndarray]:
        """
        综合权重计算
        
        Args:
            data: 数据矩阵
            y: 目标变量（用于机器学习算法）
            expert_scores: 专家打分（可选）
            
        Returns:
            综合权重结果
        """
        logger.info("开始综合权重计算...")
        
        n_features = data.shape[1]
        weights = {}
        
        # 1. 熵权法权重
        entropy_weights = self.calculate_entropy_weights(data)
        weights['entropy'] = entropy_weights
        
        # 2. 机器学习算法权重（如果有目标变量）
        if y is not None:
            ml_weights = self.calculate_ml_weights(data, y)
            weights.update(ml_weights)
        
        # 3. 专家打分法权重（如果有专家打分）
        if expert_scores is not None:
            expert_weights = self.calculate_expert_weights(expert_scores)
            weights['expert'] = expert_weights
        
        # 4. 计算最终综合权重
        if y is not None and expert_scores is not None:
            # 四种方法都有
            final_weights = (
                self.method_weights['entropy'] * entropy_weights +
                self.method_weights['ml'] * ml_weights['combined'] +
                self.method_weights['expert'] * expert_weights
            )
            # AHP权重需要单独计算，这里用简化方法
            ahp_weights = np.ones(n_features) / n_features  # 等权重作为示例
            final_weights += self.method_weights['ahp'] * ahp_weights
        elif y is not None:
            # 有机器学习权重
            final_weights = (
                self.method_weights['entropy'] * entropy_weights +
                self.method_weights['ml'] * ml_weights['combined']
            )
        else:
            # 只有熵权法
            final_weights = entropy_weights
        
        weights['final'] = final_weights
        
        logger.info("综合权重计算完成")
        return weights
    
    def get_dimension_factor_weights(self) -> Dict[str, Dict[str, float]]:
        """
        获取各维度内部因子权重
        
        Returns:
            维度因子权重字典
        """
        return {
            'compliance': {
                # 资质合规类因子（优化权重分配）
                'license_validity': 0.15,
                'fire_safety': 0.20,
                'penalty_count': 0.15,
                'warning_count': 0.10,
                'inspection_pass_rate': 0.15,
                'rectification_rate': 0.10,
                'lawsuit_rate': 0.05,
                'lawsuit_loss_rate': 0.05,
                'contract_performance': 0.05
            },
            'service_quality': {
                # 服务质量因子（优化权重分配）
                'accident_rate': 0.08,
                'accident_satisfaction': 0.06,
                'emergency_response': 0.05,
                'medical_error_rate': 0.06,
                'nursing_accident_rate': 0.05,
                'care_plan_execution': 0.08,
                'health_monitoring': 0.08,
                'service_standard': 0.08,
                'service_completion': 0.08,
                'medical_accessibility': 0.06,
                'complaint_resolution': 0.08,
                'complaint_repeat': 0.05,
                'complaint_response_time': 0.05,
                'complaint_satisfaction': 0.06,
                'elder_satisfaction': 0.10,
                'family_satisfaction': 0.08,
                'long_term_stay_rate': 0.06,
                'transfer_rate': 0.05,
                'service_quality_score': 0.08
            },
            'performance': {
                # 履约能力因子（优化权重分配）
                'revenue_growth': 0.12,
                'debt_ratio': 0.15,
                'fund_proportion': 0.10,
                'prepayment_compliance': 0.08,
                'social_security_rate': 0.08,
                'cash_flow': 0.10,
                'asset_turnover': 0.08,
                'bed_utilization': 0.10,
                'nurse_ratio': 0.08,
                'medical_staff_ratio': 0.06,
                'equipment_condition': 0.08,
                'it_level': 0.06,
                'partner_stability': 0.05,
                'management_stability': 0.05,
                'staff_turnover': 0.08,
                'service_interruption': 0.05,
                'service_recovery': 0.05,
                'strategic_planning': 0.06
            }
        }
    
    def calculate_dimension_score(self, 
                                dimension: str, 
                                factor_scores: Dict[str, float]) -> float:
        """
        计算维度得分
        
        Args:
            dimension: 维度名称
            factor_scores: 因子得分字典
            
        Returns:
            维度得分
        """
        factor_weights = self.get_dimension_factor_weights()
        
        if dimension not in factor_weights:
            raise ValueError(f"未知维度: {dimension}")
        
        dimension_score = 0.0
        for factor, weight in factor_weights[dimension].items():
            if factor in factor_scores:
                dimension_score += weight * factor_scores[factor]
        
        return dimension_score
    
    def calculate_final_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        计算最终信用评分
        
        Args:
            dimension_scores: 各维度得分
            
        Returns:
            最终信用评分
        """
        final_score = 0.0
        for dimension, score in dimension_scores.items():
            if dimension in self.dimension_weights:
                final_score += self.dimension_weights[dimension] * score
        
        return final_score


if __name__ == "__main__":
    # 测试代码
    print("权重计算模块加载成功")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.rand(100, 10)  # 100个样本，10个特征
    test_y = np.random.randint(0, 2, 100)  # 二分类目标变量
    
    # 测试权重计算
    calculator = WeightCalculator()
    
    # 计算综合权重
    weights = calculator.calculate_comprehensive_weights(test_data, test_y)
    
    print("权重计算测试完成")
    print(f"最终权重: {weights['final']}")
    print(f"权重和: {np.sum(weights['final'])}")
