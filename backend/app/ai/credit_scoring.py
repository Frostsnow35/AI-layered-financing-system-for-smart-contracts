"""
信用评分计算模块
实现基于七大维度110个影响因子的信用评分计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import math
from weight_calculator import WeightCalculator

logger = logging.getLogger(__name__)


class CreditScoringCalculator:
    """
    信用评分计算器类
    实现基于七大维度110个影响因子的信用评分计算
    """
    
    def __init__(self):
        """初始化信用评分计算器"""
        self.weight_calculator = WeightCalculator()
        
        # 七大维度权重（进一步优化权重分配）
        self.dimension_weights = {
            'compliance': 0.30,      # 合规经营维度
            'service_quality': 0.45, # 服务质量维度（提高权重）
            'performance': 0.25,     # 履约能力维度
            'social_evaluation': 0.0, # 社会评价维度（暂时无数据）
            'risk_resistance': 0.0,   # 风险抵御维度（暂时无数据）
            'innovation': 0.0,         # 创新与可持续维度（暂时无数据）
            'special_scenario': 0.0   # 特殊场景补充因子（暂时无数据）
        }
        
        # 因子评分函数参数
        self.scoring_params = self._init_scoring_params()
        
        # 非线性映射配置
        self.nonlinear_config = self._init_nonlinear_config()
    
    def _init_scoring_params(self) -> Dict[str, Dict[str, Union[float, bool]]]:
        """
        初始化评分函数参数
        
        Returns:
            评分参数字典
        """
        return {
            # 合规经营维度参数（放宽评分范围，让优秀值能得到更高分数）
            'license_validity': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'fire_safety': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'penalty_count': {'min_val': 0, 'max_val': 10, 'is_positive': False},
            'warning_count': {'min_val': 0, 'max_val': 20, 'is_positive': False},
            'inspection_pass_rate': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'rectification_rate': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'lawsuit_rate': {'min_val': 0, 'max_val': 0.3, 'is_positive': False},
            'lawsuit_loss_rate': {'min_val': 0, 'max_val': 0.5, 'is_positive': False},
            'contract_performance': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            
            # 服务质量维度参数（放宽评分范围）
            'accident_rate': {'min_val': 0, 'max_val': 0.15, 'is_positive': False},
            'accident_satisfaction': {'min_val': 70, 'max_val': 100, 'is_positive': True},
            'emergency_response': {'min_val': 1, 'max_val': 20, 'is_positive': False},  # 分钟
            'medical_error_rate': {'min_val': 0, 'max_val': 0.04, 'is_positive': False},
            'nursing_accident_rate': {'min_val': 0, 'max_val': 0.025, 'is_positive': False},
            'care_plan_execution': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'health_monitoring': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'service_standard': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'service_completion': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'medical_accessibility': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'complaint_resolution': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'complaint_repeat': {'min_val': 0, 'max_val': 0.2, 'is_positive': False},
            'complaint_response_time': {'min_val': 1, 'max_val': 7, 'is_positive': False},  # 天
            'complaint_satisfaction': {'min_val': 70, 'max_val': 100, 'is_positive': True},
            'elder_satisfaction': {'min_val': 70, 'max_val': 100, 'is_positive': True},
            'family_satisfaction': {'min_val': 70, 'max_val': 100, 'is_positive': True},
            'long_term_stay_rate': {'min_val': 0.6, 'max_val': 1.0, 'is_positive': True},
            'transfer_rate': {'min_val': 0, 'max_val': 0.15, 'is_positive': False},
            'service_quality_score': {'min_val': 70, 'max_val': 100, 'is_positive': True},
            
            # 履约能力维度参数（放宽评分范围）
            'revenue_growth': {'min_val': -0.2, 'max_val': 0.4, 'is_positive': True},
            'debt_ratio': {'min_val': 0.2, 'max_val': 0.8, 'is_positive': False},
            'fund_proportion': {'min_val': 0.5, 'max_val': 1.0, 'is_positive': True},
            'prepayment_compliance': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'social_security_rate': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'cash_flow': {'min_val': -50, 'max_val': 200, 'is_positive': True},  # 万元
            'asset_turnover': {'min_val': 0.5, 'max_val': 2.5, 'is_positive': True},
            'bed_utilization': {'min_val': 0.6, 'max_val': 1.0, 'is_positive': True},
            'nurse_ratio': {'min_val': 0.15, 'max_val': 0.4, 'is_positive': True},
            'medical_staff_ratio': {'min_val': 0.08, 'max_val': 0.25, 'is_positive': True},
            'equipment_condition': {'min_val': 0.7, 'max_val': 1.0, 'is_positive': True},
            'it_level': {'min_val': 60, 'max_val': 100, 'is_positive': True},
            'partner_stability': {'min_val': 1, 'max_val': 10, 'is_positive': True},  # 年
            'management_stability': {'min_val': 1, 'max_val': 12, 'is_positive': True},  # 年
            'staff_turnover': {'min_val': 0.05, 'max_val': 0.3, 'is_positive': False},
            'service_interruption': {'min_val': 0, 'max_val': 8, 'is_positive': False},  # 次/年
            'service_recovery': {'min_val': 1, 'max_val': 20, 'is_positive': False},  # 小时
            'strategic_planning': {'min_val': 0.6, 'max_val': 1.0, 'is_positive': True}
        }
    
    def _init_nonlinear_config(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        初始化非线性映射配置（重新设计，确保风险逻辑正确）
        
        Returns:
            非线性映射配置字典
        """
        return {
            # 风险因子使用二次函数映射，对极端值更敏感
            'penalty_count': {'type': 'quadratic'},
            'warning_count': {'type': 'quadratic'},
            'lawsuit_rate': {'type': 'quadratic'},
            'accident_rate': {'type': 'quadratic'},
            'staff_turnover': {'type': 'quadratic'},
            
            # 满意度指标使用幂函数映射，指数>1时增长加速
            'elder_satisfaction': {'type': 'power', 'exponent': 1.5},
            'family_satisfaction': {'type': 'power', 'exponent': 1.5},
            'complaint_satisfaction': {'type': 'power', 'exponent': 1.5},
            
            # 合规指标使用对数映射，体现边际效应递减
            'inspection_pass_rate': {'type': 'logarithmic'},
            'contract_performance': {'type': 'logarithmic'},
            
            # 财务指标使用幂函数映射
            'revenue_growth': {'type': 'power', 'exponent': 0.8},
            'debt_ratio': {'type': 'quadratic'},
        }
    
    def _apply_nonlinear_mapping(self, factor_name: str, normalized_value: float, is_positive: bool) -> float:
        """
        应用纯数学非线性映射（重新设计，确保风险逻辑正确）
        
        Args:
            factor_name: 因子名称
            normalized_value: 归一化值 (0-1)
            is_positive: 是否为正向指标
            
        Returns:
            非线性映射后的评分 (0-100)
        """
        if factor_name not in self.nonlinear_config:
            # 没有配置非线性映射的因子使用线性映射
            return normalized_value * 100
        
        config = self.nonlinear_config[factor_name]
        mapping_type = config['type']
        
        if mapping_type == 'logarithmic':
            # 对数映射：使用自然对数，确保单调性
            if is_positive:
                # 正向指标：使用 ln(1+x) 形式，x=0时score=0，x=1时score=100
                score = 100 * math.log(1 + normalized_value) / math.log(2)
            else:
                # 负向指标：使用 ln(2-x) 形式，x=0时score=100，x=1时score=0
                score = 100 * math.log(2 - normalized_value) / math.log(2)
        
        elif mapping_type == 'power':
            # 幂函数映射：调整敏感度
            exponent = config['exponent']
            if is_positive:
                # 正向指标：使用幂函数增长
                score = 100 * (normalized_value ** exponent)
            else:
                # 负向指标：使用幂函数衰减
                score = 100 * (1 - (normalized_value ** exponent))
        
        elif mapping_type == 'sqrt':
            # 平方根映射：边际效应递减
            if is_positive:
                # 正向指标：使用平方根增长
                score = 100 * math.sqrt(normalized_value)
            else:
                # 负向指标：使用平方根衰减
                score = 100 * (1 - math.sqrt(normalized_value))
        
        elif mapping_type == 'quadratic':
            # 二次函数映射：对极端值更敏感
            if is_positive:
                # 正向指标：使用二次函数增长
                score = 100 * (normalized_value ** 2)
            else:
                # 负向指标：使用二次函数衰减，确保单调性
                score = 100 * (1 - (normalized_value ** 2))
        
        else:
            # 默认线性映射
            score = normalized_value * 100
        
        return max(0, min(100, score))
    
    def calculate_factor_score(self, factor_name: str, value: float, use_nonlinear: bool = True) -> float:
        """
        计算单个因子得分（支持线性和非线性映射）
        
        Args:
            factor_name: 因子名称
            value: 因子值
            use_nonlinear: 是否使用非线性映射
            
        Returns:
            因子得分（0-100分）
        """
        if factor_name not in self.scoring_params:
            logger.warning(f"未知因子: {factor_name}，使用默认评分")
            return 50.0  # 默认中等分数
        
        params = self.scoring_params[factor_name]
        min_val = params['min_val']
        max_val = params['max_val']
        is_positive = params['is_positive']
        
        # 确保值在合理范围内
        value = max(min_val, min(max_val, value))
        
        # 归一化到0-1范围
        if is_positive:
            # 正向指标：值越大越好
            normalized_value = (value - min_val) / (max_val - min_val)
        else:
            # 负向指标：值越小越好
            normalized_value = (max_val - value) / (max_val - min_val)
        
        # 应用映射函数
        if use_nonlinear:
            # 对于非线性映射，我们需要重新归一化
            if is_positive:
                # 正向指标：直接归一化
                normalized_value = (value - min_val) / (max_val - min_val)
            else:
                # 负向指标：直接归一化（值越大风险越高）
                normalized_value = (value - min_val) / (max_val - min_val)
            score = self._apply_nonlinear_mapping(factor_name, normalized_value, is_positive)
        else:
            # 线性映射
            score = normalized_value * 100
        
        return max(0, min(100, score))
    
    def _apply_non_profit_bonus(self, dimension_data: Dict[str, Dict[str, float]], base_score: float) -> float:
        """
        应用非营利机构特殊加分机制
        
        Args:
            dimension_data: 各维度数据字典
            base_score: 基础评分
            
        Returns:
            调整后的评分
        """
        bonus_score = 0
        
        # 检查是否为非营利机构（基于政府补贴占比等指标）
        performance_data = dimension_data.get('performance', {})
        fund_proportion = performance_data.get('fund_proportion', 0)
        
        # 非营利机构特征识别
        is_non_profit = False
        
        # 1. 政府补贴占比高（≥60%）
        if fund_proportion >= 0.6:
            is_non_profit = True
            bonus_score += 8  # 政府补贴稳定性加分
        
        # 2. 入住率高（≥85%）
        bed_utilization = performance_data.get('bed_utilization', 0)
        if bed_utilization >= 0.85:
            bonus_score += 5  # 高入住率加分
        
        # 3. 安全事故零发生
        service_data = dimension_data.get('service_quality', {})
        accident_rate = service_data.get('accident_rate', 0)
        if accident_rate == 0:
            bonus_score += 3  # 零事故加分
        
        # 4. 现金流质量好（≥90%）
        cash_flow = performance_data.get('cash_flow', 0)
        if cash_flow >= 90:
            bonus_score += 4  # 现金流质量加分
        
        # 5. 满意度高（≥90分）
        elder_satisfaction = service_data.get('elder_satisfaction', 0)
        if elder_satisfaction >= 90:
            bonus_score += 3  # 高满意度加分
        
        # 6. 合规性优秀（≥95%）
        compliance_data = dimension_data.get('compliance', {})
        license_validity = compliance_data.get('license_validity', 0)
        if license_validity >= 0.95:
            bonus_score += 2  # 合规性加分
        
        # 7. 医养结合优势（基于真实案例）
        medical_staff_ratio = performance_data.get('medical_staff_ratio', 0)
        if medical_staff_ratio >= 0.15:  # 医疗人员比例≥15%
            bonus_score += 3  # 医养结合加分
        
        # 应用加分
        adjusted_score = base_score + bonus_score
        
        # 确保评分不超过100分
        final_score = min(100, adjusted_score)
        
        if bonus_score > 0:
            logger.info(f"非营利机构特殊加分: +{bonus_score:.1f}分 (基础分: {base_score:.1f} -> 调整后: {final_score:.1f})")
        
        return final_score
    
    def calculate_dimension_score(self, 
                                dimension: str, 
                                factor_values: Dict[str, float], 
                                use_nonlinear: bool = True) -> float:
        """
        计算维度得分
        
        Args:
            dimension: 维度名称
            factor_values: 因子值字典
            use_nonlinear: 是否使用非线性映射
            
        Returns:
            维度得分（0-100分）
        """
        factor_weights = self.weight_calculator.get_dimension_factor_weights()
        
        if dimension not in factor_weights:
            logger.warning(f"未知维度: {dimension}")
            return 50.0
        
        dimension_score = 0.0
        total_weight = 0.0
        
        for factor, weight in factor_weights[dimension].items():
            if factor in factor_values:
                factor_score = self.calculate_factor_score(factor, factor_values[factor], use_nonlinear)
                dimension_score += weight * factor_score
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            dimension_score = dimension_score / total_weight
        
        return dimension_score
    
    def calculate_credit_score(self, 
                             dimension_data: Dict[str, Dict[str, float]], 
                             use_nonlinear: bool = True) -> Dict[str, Union[float, str]]:
        """
        计算信用评分（支持线性和非线性映射）
        
        Args:
            dimension_data: 各维度数据字典
            格式: {
                'compliance': {'license_validity': 0.95, 'fire_safety': 0.90, ...},
                'service_quality': {'accident_rate': 0.02, 'elder_satisfaction': 85, ...},
                ...
            }
            use_nonlinear: 是否使用非线性映射
            
        Returns:
            信用评分结果
        """
        mapping_type = "非线性" if use_nonlinear else "线性"
        logger.info(f"开始计算信用评分（{mapping_type}映射）...")
        
        dimension_scores = {}
        
        # 计算各维度得分
        for dimension, factor_values in dimension_data.items():
            dimension_score = self.calculate_dimension_score(dimension, factor_values, use_nonlinear)
            dimension_scores[dimension] = dimension_score
            logger.info(f"{dimension}维度得分: {dimension_score:.2f}")
        
        # 计算最终信用评分
        final_score = 0.0
        for dimension, score in dimension_scores.items():
            if dimension in self.dimension_weights:
                final_score += self.dimension_weights[dimension] * score
        
        # 非营利机构特殊加分机制
        final_score = self._apply_non_profit_bonus(dimension_data, final_score)
        
        # 确定信用等级
        credit_level = self._get_credit_level(final_score)
        
        # 计算违约概率（简化计算）
        default_probability = max(0, min(1, (100 - final_score) / 100))
        
        result = {
            'credit_score': final_score,
            'credit_level': credit_level,
            'default_probability': default_probability,
            'dimension_scores': dimension_scores,
            'dimension_weights': self.dimension_weights,
            'mapping_type': mapping_type
        }
        
        logger.info(f"信用评分计算完成: {final_score:.2f}分 ({credit_level})")
        
        return result
    
    def _get_credit_level(self, score: float) -> str:
        """
        根据评分确定信用等级（优化后的评分标准）
        
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
    
    def calculate_risk_warning(self, 
                             dimension_data: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """
        计算风险预警
        
        Args:
            dimension_data: 各维度数据字典
            
        Returns:
            风险预警结果
        """
        warnings = {}
        
        # 定义风险阈值
        risk_thresholds = {
            'compliance': {
                'penalty_count': 2,  # 罚款次数超过2次
                'warning_count': 5,  # 警告次数超过5次
                'lawsuit_rate': 0.1,  # 涉诉率超过10%
                'inspection_pass_rate': 0.8  # 检查合格率低于80%
            },
            'service_quality': {
                'accident_rate': 0.05,  # 事故率超过5%
                'complaint_resolution': 0.9,  # 投诉解决率低于90%
                'elder_satisfaction': 80,  # 老人满意度低于80分
                'emergency_response': 10  # 紧急响应时间超过10分钟
            },
            'performance': {
                'debt_ratio': 0.7,  # 资产负债率超过70%
                'revenue_growth': 0,  # 营业收入增长率为负
                'bed_utilization': 0.5,  # 床位利用率低于50%
                'staff_turnover': 0.2  # 员工流失率超过20%
            }
        }
        
        for dimension, thresholds in risk_thresholds.items():
            dimension_warnings = []
            
            if dimension in dimension_data:
                for factor, threshold in thresholds.items():
                    if factor in dimension_data[dimension]:
                        value = dimension_data[dimension][factor]
                        
                        # 判断是否触发预警
                        if factor in ['penalty_count', 'warning_count', 'lawsuit_rate', 'accident_rate', 'staff_turnover']:
                            # 负向指标：值超过阈值
                            if value > threshold:
                                dimension_warnings.append(f"{factor}: {value} > {threshold}")
                        else:
                            # 正向指标：值低于阈值
                            if value < threshold:
                                dimension_warnings.append(f"{factor}: {value} < {threshold}")
            
            if dimension_warnings:
                warnings[dimension] = dimension_warnings
        
        return warnings
    
    def generate_score_report(self, 
                            dimension_data: Dict[str, Dict[str, float]]) -> str:
        """
        生成评分报告
        
        Args:
            dimension_data: 各维度数据字典
            
        Returns:
            评分报告文本
        """
        # 计算信用评分
        score_result = self.calculate_credit_score(dimension_data)
        
        # 计算风险预警
        warnings = self.calculate_risk_warning(dimension_data)
        
        # 生成报告
        report = f"""
=== 养老服务提供商信用评分报告 ===

【总体评分】
信用评分: {score_result['credit_score']:.2f}分
信用等级: {score_result['credit_level']}
违约概率: {score_result['default_probability']:.2%}

【各维度评分】
"""
        
        for dimension, score in score_result['dimension_scores'].items():
            weight = score_result['dimension_weights'].get(dimension, 0)
            report += f"{dimension}: {score:.2f}分 (权重: {weight:.1%})\n"
        
        # 添加风险预警
        if warnings:
            report += "\n【风险预警】\n"
            for dimension, dimension_warnings in warnings.items():
                report += f"{dimension}:\n"
                for warning in dimension_warnings:
                    report += f"  - {warning}\n"
        else:
            report += "\n【风险预警】\n无风险预警\n"
        
        report += "\n=== 报告结束 ==="
        
        return report
    
    def batch_calculate_scores(self, 
                              batch_data: List[Dict[str, Dict[str, float]]]) -> List[Dict[str, Union[float, str]]]:
        """
        批量计算信用评分
        
        Args:
            batch_data: 批量数据列表
            
        Returns:
            批量评分结果列表
        """
        logger.info(f"开始批量计算信用评分，共{len(batch_data)}个样本")
        
        results = []
        for i, dimension_data in enumerate(batch_data):
            try:
                result = self.calculate_credit_score(dimension_data)
                results.append(result)
                logger.info(f"样本{i+1}评分完成: {result['credit_score']:.2f}分")
            except Exception as e:
                logger.error(f"样本{i+1}评分失败: {str(e)}")
                results.append({
                    'credit_score': 0.0,
                    'credit_level': 'E级（差信用，风险很高）',
                    'default_probability': 1.0,
                    'error': str(e)
                })
        
        logger.info("批量计算完成")
        return results


if __name__ == "__main__":
    # 测试代码
    print("信用评分计算模块加载成功")
    
    # 创建测试数据
    test_data = {
        'compliance': {
            'license_validity': 0.95,
            'fire_safety': 0.90,
            'penalty_count': 0,
            'warning_count': 1,
            'inspection_pass_rate': 0.95,
            'rectification_rate': 0.90,
            'lawsuit_rate': 0.02,
            'lawsuit_loss_rate': 0.1,
            'contract_performance': 0.95
        },
        'service_quality': {
            'accident_rate': 0.02,
            'accident_satisfaction': 85,
            'emergency_response': 5,
            'medical_error_rate': 0.01,
            'nursing_accident_rate': 0.005,
            'care_plan_execution': 0.90,
            'health_monitoring': 0.85,
            'service_standard': 0.88,
            'service_completion': 0.92,
            'medical_accessibility': 0.90,
            'complaint_resolution': 0.95,
            'complaint_repeat': 0.05,
            'complaint_response_time': 2,
            'complaint_satisfaction': 88,
            'elder_satisfaction': 85,
            'family_satisfaction': 82,
            'long_term_stay_rate': 0.75,
            'transfer_rate': 0.05,
            'service_quality_score': 85
        },
        'performance': {
            'revenue_growth': 0.08,
            'debt_ratio': 0.45,
            'fund_proportion': 0.65,
            'prepayment_compliance': 0.95,
            'social_security_rate': 0.90,
            'cash_flow': 50,
            'asset_turnover': 1.2,
            'bed_utilization': 0.75,
            'nurse_ratio': 0.25,
            'medical_staff_ratio': 0.15,
            'equipment_condition': 0.85,
            'it_level': 75,
            'partner_stability': 3,
            'management_stability': 5,
            'staff_turnover': 0.15,
            'service_interruption': 2,
            'service_recovery': 4,
            'strategic_planning': 0.80
        }
    }
    
    # 测试评分计算
    calculator = CreditScoringCalculator()
    result = calculator.calculate_credit_score(test_data)
    
    print("信用评分测试完成")
    print(f"信用评分: {result['credit_score']:.2f}分")
    print(f"信用等级: {result['credit_level']}")
    print(f"违约概率: {result['default_probability']:.2%}")
    
    # 生成报告
    report = calculator.generate_score_report(test_data)
    print("\n评分报告:")
    print(report)
