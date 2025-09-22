# AI+智能合约分层融资系统 - 核心模块2

专注于AI+智能合约的分层融资系统，实现养老服务机构的精准授信和自动履约，覆盖短期流动资金和中长期发展资金需求。

## 项目概述

本项目专注于实现核心模块2：AI+智能合约的分层融资系统。通过AI风险评分模型和智能合约技术，为养老服务机构提供精准授信和自动履约服务，大幅提升融资效率和降低风险。

### 核心功能

- **非抵押信贷融资**: AI风险评分 + 智能合约自动放款
- **私募小额债权融资**: AI智能匹配 + 智能合约资金监控  
- **自动履约系统**: 到期自动扣款，违约自动处置

### 技术架构

- **后端**: FastAPI + Python 3.9+
- **前端**: Vue 3 + Element Plus + TypeScript
- **区块链**: FISCO BCOS 3.0
- **数据库**: SQLite 3 (轻量化部署)
- **AI/ML**: Scikit-learn + Pandas + NumPy
- **智能合约**: Solidity 0.8+

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- FISCO BCOS 3.0 (可选)

### 快速启动

#### 一键启动（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd 分层融资系统

# 一键启动核心模块2
./scripts/start_core2.sh
```

#### 手动启动

```bash
# 1. 环境配置
cp env.example .env

# 2. 启动后端
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/main.py

# 3. 启动前端
cd frontend/web
npm install
npm run dev
```

#### AI模型训练

```bash
# 训练风险评分模型
python scripts/train_model.py --data-size 5000 --model-path ./models
```

## 项目结构

```
分层融资系统/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── api/            # API路由
│   │   ├── core/           # 核心配置
│   │   ├── models/         # 数据模型
│   │   ├── services/       # 业务逻辑
│   │   ├── utils/          # 工具函数
│   │   └── ai/             # AI模型
│   ├── blockchain/         # 区块链交互
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端应用
│   └── web/               # Web管理界面
├── blockchain/             # 区块链配置
│   ├── scripts/           # 部署脚本
│   └── contracts/         # 智能合约源码
│       ├── CreditContract.sol      # 非抵押信贷合约
│       └── FundMatchingContract.sol # 私募债权合约
├── scripts/               # 部署脚本
│   ├── start_core2.sh     # 启动脚本
│   ├── stop_core2.sh      # 停止脚本
│   └── train_model.py     # AI模型训练
└── README.md
```

## 核心功能说明

### 1. 非抵押信贷融资
- **AI风险评分**: 基于服务验收通过率(40%) + 政府补贴到账率(30%) + 应收账款逾期率(30%)
- **智能合约**: 自动验证资产确权凭证，AI评分，自动放款
- **自动履约**: 到期从服务收入/政府补贴中自动扣款

### 2. 私募小额债权融资  
- **AI智能匹配**: 基于企业需求和基金偏好进行智能匹配
- **资金监控**: 实时监控资金流向，确保专款专用
- **自动履约**: 到期自动履约，违约触发资产处置

### 3. 预期效果
- 融资审批时间从12个工作日缩短至2小时（效率提升97%）
- 民办非营利养老机构融资成功率从28.7%提升至65%以上
- AI模型准确率达89.2%，私募基金匹配准确率达82%

## API 文档

启动后端服务后，访问以下地址查看 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目链接: [项目地址]

## 致谢

感谢以下开源项目的支持：

- [FastAPI](https://fastapi.tiangolo.com/)
- [Vue.js](https://vuejs.org/)
- [FISCO BCOS](https://fisco-bcos-documentation.readthedocs.io/)
- [Element Plus](https://element-plus.org/)
