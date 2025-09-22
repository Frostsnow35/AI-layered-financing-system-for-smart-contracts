#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于项目企划.txt和行业真实数据生成典型养老服务测试数据
参考民政部、中国银行业协会等行业标准
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List
import json

class RealisticDataGenerator:
    """基于行业真实数据的养老服务测试数据生成器"""
    
    def __init__(self):
        """初始化数据生成器"""
        # 基于项目企划.txt中的行业数据设置参数
        self.industry_params = {
            # 民办非营利机构融资成功率：28.7%（民政部2023年数据）
            'financing_success_rate': 0.287,
            # 行业整体盈利能力：-15.5亿元（中国养老金融50人论坛2023年）
            'industry_profitability': -0.155,
            # 应收账款占比：60%（中国物流与采购联合会2024年）
            'receivables_ratio': 0.6,
            # 账期：90-180天
            'payment_terms': (90, 180),
            # 融资成本比大型企业高2-4个百分点
            'financing_cost_premium': (0.02, 0.04),
            # 养老服务概念股占比：2.21%（Wind金融终端2024年6月）
            'listed_ratio': 0.0221,
            # 私募养老基金投资覆盖率：17%（中基协2024年）
            'private_fund_coverage': 0.17
        }
        
        # 基于民政部标准的风险阈值
        self.risk_thresholds = {
            'receivables_overdue_rate': 0.08,  # 应收账款逾期率≥8%
            'service_acceptance_overdue': 0.10,  # 服务验收逾期率≥10%
            'revenue_volatility': 0.20,  # 服务收入波动≥20%
            'staff_turnover_rate': 0.15,  # 人员流失率≥15%
            'debt_ratio': 0.6,  # 负债率≥60%
            'cash_flow_ratio': 0.1  # 现金流比率≥10%
        }
    
    def generate_enterprise_data(self, enterprise_type: str, risk_level: str) -> Dict:
        """
        生成企业级养老服务数据
        
        Args:
            enterprise_type: 企业类型 ('non_profit', 'for_profit', 'small_enterprise')
            risk_level: 风险等级 ('low', 'medium', 'high', 'severe')
            
        Returns:
            企业数据字典
        """
        base_data = self._get_base_data_by_type(enterprise_type)
        risk_adjustments = self._get_risk_adjustments(risk_level)
        
        # 应用风险调整
        adjusted_data = self._apply_risk_adjustments(base_data, risk_adjustments)
        
        return adjusted_data
    
    def _get_base_data_by_type(self, enterprise_type: str) -> Dict:
        """根据企业类型获取基础数据"""
        if enterprise_type == 'non_profit':
            # 民办非营利养老机构（占行业62%）
            return {
                'compliance': {
                    'license_validity': 0.95,  # 非营利机构合规性较高
                    'fire_safety': 0.92,
                    'penalty_count': 0,  # 非营利机构罚款较少
                    'warning_count': 1,
                    'inspection_pass_rate': 0.88,  # 民政部门监管严格
                    'rectification_rate': 0.85,
                    'lawsuit_rate': 0.02,  # 非营利机构诉讼率较低
                    'lawsuit_loss_rate': 0.05,
                    'contract_performance': 0.90
                },
                'service_quality': {
                    'accident_rate': 0.015,  # 非营利机构事故率较低
                    'accident_satisfaction': 88,
                    'emergency_response': 6,
                    'medical_error_rate': 0.008,
                    'nursing_accident_rate': 0.003,
                    'care_plan_execution': 0.88,
                    'health_monitoring': 0.90,
                    'service_standard': 0.92,
                    'service_completion': 0.89,
                    'medical_accessibility': 0.87,
                    'complaint_resolution': 0.91,
                    'complaint_repeat': 0.03,
                    'complaint_response_time': 2.5,
                    'complaint_satisfaction': 89,
                    'elder_satisfaction': 91,
                    'family_satisfaction': 88,
                    'long_term_stay_rate': 0.85,
                    'transfer_rate': 0.04,
                    'service_quality_score': 89
                },
                'performance': {
                    'revenue_growth': 0.08,  # 非营利机构增长稳定
                    'debt_ratio': 0.25,  # 负债率较低
                    'fund_proportion': 0.75,  # 政府补贴占比高
                    'prepayment_compliance': 0.88,
                    'social_security_rate': 0.95,
                    'cash_flow': 45,  # 现金流相对稳定
                    'asset_turnover': 1.8,
                    'bed_utilization': 0.82,
                    'nurse_ratio': 0.28,
                    'medical_staff_ratio': 0.18,
                    'equipment_condition': 0.85,
                    'it_level': 78,
                    'partner_stability': 9,
                    'management_stability': 11,
                    'staff_turnover': 0.08,  # 人员流失率较低
                    'service_interruption': 1,
                    'service_recovery': 3,
                    'strategic_planning': 0.88
                }
            }
        
        elif enterprise_type == 'for_profit':
            # 营利性养老机构
            return {
                'compliance': {
                    'license_validity': 0.88,
                    'fire_safety': 0.85,
                    'penalty_count': 1,
                    'warning_count': 2,
                    'inspection_pass_rate': 0.82,
                    'rectification_rate': 0.80,
                    'lawsuit_rate': 0.05,
                    'lawsuit_loss_rate': 0.12,
                    'contract_performance': 0.85
                },
                'service_quality': {
                    'accident_rate': 0.025,
                    'accident_satisfaction': 82,
                    'emergency_response': 8,
                    'medical_error_rate': 0.015,
                    'nursing_accident_rate': 0.008,
                    'care_plan_execution': 0.82,
                    'health_monitoring': 0.85,
                    'service_standard': 0.88,
                    'service_completion': 0.84,
                    'medical_accessibility': 0.82,
                    'complaint_resolution': 0.86,
                    'complaint_repeat': 0.06,
                    'complaint_response_time': 3.5,
                    'complaint_satisfaction': 84,
                    'elder_satisfaction': 86,
                    'family_satisfaction': 83,
                    'long_term_stay_rate': 0.78,
                    'transfer_rate': 0.08,
                    'service_quality_score': 84
                },
                'performance': {
                    'revenue_growth': 0.15,  # 营利性机构增长较快
                    'debt_ratio': 0.45,  # 负债率适中
                    'fund_proportion': 0.35,  # 政府补贴占比低
                    'prepayment_compliance': 0.82,
                    'social_security_rate': 0.88,
                    'cash_flow': 25,  # 现金流波动较大
                    'asset_turnover': 2.2,
                    'bed_utilization': 0.88,
                    'nurse_ratio': 0.32,
                    'medical_staff_ratio': 0.22,
                    'equipment_condition': 0.88,
                    'it_level': 85,
                    'partner_stability': 8,
                    'management_stability': 9,
                    'staff_turnover': 0.12,
                    'service_interruption': 2,
                    'service_recovery': 5,
                    'strategic_planning': 0.85
                }
            }
        
        else:  # small_enterprise
            # 中小养老服务企业（康复护理公司、居家养老团队）
            return {
                'compliance': {
                    'license_validity': 0.82,
                    'fire_safety': 0.80,
                    'penalty_count': 2,
                    'warning_count': 3,
                    'inspection_pass_rate': 0.78,
                    'rectification_rate': 0.75,
                    'lawsuit_rate': 0.08,
                    'lawsuit_loss_rate': 0.18,
                    'contract_performance': 0.78
                },
                'service_quality': {
                    'accident_rate': 0.035,
                    'accident_satisfaction': 78,
                    'emergency_response': 12,
                    'medical_error_rate': 0.025,
                    'nursing_accident_rate': 0.015,
                    'care_plan_execution': 0.75,
                    'health_monitoring': 0.78,
                    'service_standard': 0.80,
                    'service_completion': 0.76,
                    'medical_accessibility': 0.75,
                    'complaint_resolution': 0.80,
                    'complaint_repeat': 0.10,
                    'complaint_response_time': 5,
                    'complaint_satisfaction': 79,
                    'elder_satisfaction': 81,
                    'family_satisfaction': 78,
                    'long_term_stay_rate': 0.70,
                    'transfer_rate': 0.12,
                    'service_quality_score': 79
                },
                'performance': {
                    'revenue_growth': 0.05,  # 中小企业增长较慢
                    'debt_ratio': 0.55,  # 负债率较高
                    'fund_proportion': 0.25,  # 政府补贴占比低
                    'prepayment_compliance': 0.75,
                    'social_security_rate': 0.80,
                    'cash_flow': 8,  # 现金流紧张
                    'asset_turnover': 1.5,
                    'bed_utilization': 0.75,
                    'nurse_ratio': 0.25,
                    'medical_staff_ratio': 0.15,
                    'equipment_condition': 0.75,
                    'it_level': 70,
                    'partner_stability': 6,
                    'management_stability': 7,
                    'staff_turnover': 0.18,  # 人员流失率高
                    'service_interruption': 4,
                    'service_recovery': 8,
                    'strategic_planning': 0.75
                }
            }
    
    def _get_risk_adjustments(self, risk_level: str) -> Dict:
        """根据风险等级获取调整系数"""
        adjustments = {
            'low': {
                'multiplier': 1.0,
                'penalty_adjustment': 0,
                'satisfaction_adjustment': 5,
                'debt_adjustment': -0.05
            },
            'medium': {
                'multiplier': 0.9,
                'penalty_adjustment': 1,
                'satisfaction_adjustment': -3,
                'debt_adjustment': 0.05
            },
            'high': {
                'multiplier': 0.7,
                'penalty_adjustment': 3,
                'satisfaction_adjustment': -8,
                'debt_adjustment': 0.15
            },
            'severe': {
                'multiplier': 0.5,
                'penalty_adjustment': 5,
                'satisfaction_adjustment': -15,
                'debt_adjustment': 0.25
            }
        }
        return adjustments[risk_level]
    
    def _apply_risk_adjustments(self, base_data: Dict, adjustments: Dict) -> Dict:
        """应用风险调整"""
        adjusted_data = {}
        
        for dimension, factors in base_data.items():
            adjusted_data[dimension] = {}
            for factor, value in factors.items():
                if isinstance(value, (int, float)):
                    if 'penalty' in factor or 'warning' in factor or 'lawsuit' in factor:
                        # 风险因子：值越大风险越高
                        adjusted_value = value + adjustments['penalty_adjustment']
                    elif 'satisfaction' in factor or 'elder' in factor or 'family' in factor:
                        # 满意度因子：值越大越好
                        adjusted_value = max(0, value + adjustments['satisfaction_adjustment'])
                    elif 'debt' in factor:
                        # 负债率：值越大风险越高
                        adjusted_value = min(1.0, value + adjustments['debt_adjustment'])
                    else:
                        # 其他因子：按倍数调整
                        adjusted_value = value * adjustments['multiplier']
                    
                    adjusted_data[dimension][factor] = adjusted_value
                else:
                    adjusted_data[dimension][factor] = value
        
        return adjusted_data
    
    def generate_industry_dataset(self, num_samples: int = 100) -> List[Dict]:
        """
        生成行业典型数据集
        
        Args:
            num_samples: 样本数量
            
        Returns:
            数据集列表
        """
        dataset = []
        
        # 基于项目企划.txt中的行业分布
        enterprise_types = ['non_profit', 'for_profit', 'small_enterprise']
        type_weights = [0.62, 0.25, 0.13]  # 非营利62%，营利25%，中小企业13%
        
        risk_levels = ['low', 'medium', 'high', 'severe']
        risk_weights = [0.35, 0.40, 0.20, 0.05]  # 基于行业风险分布
        
        for i in range(num_samples):
            # 按权重选择企业类型
            enterprise_type = np.random.choice(enterprise_types, p=type_weights)
            risk_level = np.random.choice(risk_levels, p=risk_weights)
            
            # 生成企业数据
            enterprise_data = self.generate_enterprise_data(enterprise_type, risk_level)
            
            # 添加元数据
            enterprise_data['metadata'] = {
                'enterprise_type': enterprise_type,
                'risk_level': risk_level,
                'sample_id': i + 1,
                'generation_method': 'realistic_industry_data'
            }
            
            dataset.append(enterprise_data)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = 'realistic_industry_dataset.json'):
        """保存数据集"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"数据集已保存到: {filename}")
    
    def generate_test_scenarios(self) -> Dict[str, Dict]:
        """生成典型测试场景"""
        scenarios = {}
        
        # 1. 优秀非营利养老院（A级）
        scenarios['excellent_non_profit'] = self.generate_enterprise_data('non_profit', 'low')
        scenarios['excellent_non_profit']['metadata'] = {
            'enterprise_type': 'non_profit',
            'risk_level': 'low',
            'description': '优秀非营利养老院，政府补贴稳定，服务优质'
        }
        
        # 2. 典型营利性养老院（B级）
        scenarios['typical_for_profit'] = self.generate_enterprise_data('for_profit', 'low')
        scenarios['typical_for_profit']['metadata'] = {
            'enterprise_type': 'for_profit',
            'risk_level': 'low',
            'description': '典型营利性养老院，市场化运营，增长稳定'
        }
        
        # 3. 中等风险中小企业（C级）
        scenarios['medium_small_enterprise'] = self.generate_enterprise_data('small_enterprise', 'medium')
        scenarios['medium_small_enterprise']['metadata'] = {
            'enterprise_type': 'small_enterprise',
            'risk_level': 'medium',
            'description': '中等风险中小企业，资金链紧张但可维持'
        }
        
        # 4. 高风险营利性机构（D级）
        scenarios['high_risk_for_profit'] = self.generate_enterprise_data('for_profit', 'high')
        scenarios['high_risk_for_profit']['metadata'] = {
            'enterprise_type': 'for_profit',
            'risk_level': 'high',
            'description': '高风险营利性机构，经营困难，风险较高'
        }
        
        # 5. 严重风险中小企业（E级）
        scenarios['severe_risk_small_enterprise'] = self.generate_enterprise_data('small_enterprise', 'severe')
        scenarios['severe_risk_small_enterprise']['metadata'] = {
            'enterprise_type': 'small_enterprise',
            'risk_level': 'severe',
            'description': '严重风险中小企业，濒临破产，风险极高'
        }
        
        return scenarios

def main():
    """主函数"""
    print("🚀 开始生成基于行业真实数据的养老服务测试数据")
    print("=" * 80)
    
    # 创建数据生成器
    generator = RealisticDataGenerator()
    
    # 生成典型测试场景
    print("📊 生成典型测试场景...")
    test_scenarios = generator.generate_test_scenarios()
    
    # 保存测试场景
    generator.save_dataset(list(test_scenarios.values()), 'test_scenarios.json')
    
    # 生成行业数据集
    print("📈 生成行业典型数据集...")
    industry_dataset = generator.generate_industry_dataset(100)
    
    # 保存行业数据集
    generator.save_dataset(industry_dataset, 'industry_dataset.json')
    
    # 打印统计信息
    print("\n📋 数据生成统计:")
    print("=" * 50)
    
    type_counts = {}
    risk_counts = {}
    
    for sample in industry_dataset:
        enterprise_type = sample['metadata']['enterprise_type']
        risk_level = sample['metadata']['risk_level']
        
        type_counts[enterprise_type] = type_counts.get(enterprise_type, 0) + 1
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
    
    print("企业类型分布:")
    for enterprise_type, count in type_counts.items():
        percentage = (count / len(industry_dataset)) * 100
        print(f"  {enterprise_type}: {count}个 ({percentage:.1f}%)")
    
    print("\n风险等级分布:")
    for risk_level, count in risk_counts.items():
        percentage = (count / len(industry_dataset)) * 100
        print(f"  {risk_level}: {count}个 ({percentage:.1f}%)")
    
    print("\n🎉 数据生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
