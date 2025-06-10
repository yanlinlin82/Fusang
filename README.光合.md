# Fusang（扶桑）：基于深度学习的系统发育树分析软件

我们开发了一种新型的系统发育树推断软件Fusang，它结合了卷积神经网络（CNN）和双向长短期记忆网络（Bi-LSTM），能够适应不同长度的序列，并有效地从多序列比对（MSA）数据中提取特征，从而准确推断物种或基因之间的进化关系。这种设计使其在模拟和真实数据集上的性能表现，都能与基于最大似然的方法相媲美，同时还可以通过对不同应用场景的定制训练，来针对性地适配特定场景以进一步提升准确性，具备超越传统工具的潜力。

## 软件编译安装流程

这里描述的安装流程，是镜像构建过程。通过镜像启动容器可以跳过此步骤。

DCU版本安装：

```sh
cd /opt/
git clone https://github.com/yanlinlin82/fusang/

cd /fusang/
pip install -U pip
pip install -U pip-tools
pip-compile -c constraints.DCU.txt requirements.DCU.in -o requirements.DCU.txt
pip install -c constraints.DCU.txt -r requirements.DCU.txt
```

普通GPU（或CPU）版本安装：

```sh
cd /fusang/
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -U pip-tools
pip-compile requirements.in -o requirements.GPU.txt
pip install -r requirements.GPU.txt
```

## 算例介绍

### 算例一

算例名称及简介：哺乳动物COX1基因的系统发育树分析

在 NCBI 上搜索 "COX1[Gene] AND mammals[Organism] AND 500:1500[Sequence Length]"，可获取到 11,976 条序列。<https://www.ncbi.nlm.nih.gov/nuccore/?term=COX1%5BGene%5D+AND+mammals%5BOrganism%5D+AND+500%3A1500%5BSequence+Length%5D>

从这些序列中抽取部分序列，用于本测试。

运行指令：

```sh
cd /opt/fusang/

# DCU版本
python fusang.py --msa_dir example_msa/mammals-cox1_1.fas --save_prefix mammals-cox1_1_dcu
python fusang.py --msa_dir example_msa/mammals-cox1_2.fas --save_prefix mammals-cox1_2_dcu
python fusang.py --msa_dir example_msa/mammals-cox1_3.fas --save_prefix mammals-cox1_2_dcu

# CPU版本
.venv/bin/python fusang.py --msa_dir example_msa/mammals-cox1_1.fas --save_prefix mammals-cox1_1_cpu
.venv/bin/python fusang.py --msa_dir example_msa/mammals-cox1_2.fas --save_prefix mammals-cox1_2_cpu
.venv/bin/python fusang.py --msa_dir example_msa/mammals-cox1_3.fas --save_prefix mammals-cox1_3_cpu
```

运行结果：

```sh
./dl_output/mammals-cox1_1_dcu.txt
./dl_output/mammals-cox1_2_dcu.txt
./dl_output/mammals-cox1_3_dcu.txt

./dl_output/mammals-cox1_1_cpu.txt
./dl_output/mammals-cox1_2_cpu.txt
./dl_output/mammals-cox1_3_cpu.txt
```

正确性一致说明：预期DCU版本和CPU版本的对应结果完全相同

性能说明：

a.DCU单卡比CPU单核心加速比
b.DCU单卡比CPU 32核心加速比
c.DCU多卡/单卡 加速比及并行效率
d.此算例跟成熟软件对比，正确性和性能说明

### 算例二

算例名称及简介：线粒体编码基因细胞色素b的系统发育树分析

在 NCBI 上搜索 "CYTB[Gene] AND 500:1500[Sequence Length]"，可获取到 456,968 条序列。<https://www.ncbi.nlm.nih.gov/nuccore/?term=CYTB%5BGene%5D+AND+500%3A1500%5BSequence+Length%5D>

从这些序列中抽取部分序列，用于本测试。

运行指令：

```sh
cd /opt/fusang/

# DCU版本
python fusang.py --msa_dir example_msa/CYTB_1.fas --save_prefix CYTB_1_dcu
python fusang.py --msa_dir example_msa/CYTB_2.fas --save_prefix CYTB_2_dcu
python fusang.py --msa_dir example_msa/CYTB_3.fas --save_prefix CYTB_2_dcu

# CPU版本
.venv/bin/python fusang.py --msa_dir example_msa/CYTB_1.fas --save_prefix CYTB_1_cpu
.venv/bin/python fusang.py --msa_dir example_msa/CYTB_2.fas --save_prefix CYTB_2_cpu
.venv/bin/python fusang.py --msa_dir example_msa/CYTB_3.fas --save_prefix CYTB_3_cpu
```

运行结果：

```sh
./dl_output/CYTB_1_dcu.txt
./dl_output/CYTB_2_dcu.txt
./dl_output/CYTB_3_dcu.txt

./dl_output/CYTB_1_cpu.txt
./dl_output/CYTB_2_cpu.txt
./dl_output/CYTB_3_cpu.txt
```

正确性一致说明：预期DCU版本和CPU版本的对应结果完全相同

性能说明：

a.DCU单卡比CPU单核心加速比
b.DCU单卡比CPU 32核心加速比
c.DCU多卡/单卡 加速比及并行效率
d.此算例跟成熟软件对比，正确性和性能说明

### 算例三

算例名称及简介：新冠病毒序列的系统发育树分析

在 NCBI 的新冠病毒数据库上下载原始序列，全库有超过900万条序列，除采取全库下载外，也可每次随机下载2000条。<https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=taxid:2697049>

从这些序列中抽取部分序列，用于本测试。

运行指令：

```sh
cd /opt/fusang/

# DCU版本
python fusang.py --msa_dir example_msa/sars-cov-2_1.fas --save_prefix sars-cov-2_1_dcu
python fusang.py --msa_dir example_msa/sars-cov-2_2.fas --save_prefix sars-cov-2_2_dcu
python fusang.py --msa_dir example_msa/sars-cov-2_3.fas --save_prefix sars-cov-2_2_dcu

# CPU版本
.venv/bin/python fusang.py --msa_dir example_msa/sars-cov-2_1.fas --save_prefix sars-cov-2_1_cpu
.venv/bin/python fusang.py --msa_dir example_msa/sars-cov-2_2.fas --save_prefix sars-cov-2_2_cpu
.venv/bin/python fusang.py --msa_dir example_msa/sars-cov-2_3.fas --save_prefix sars-cov-2_3_cpu
```

运行结果：

```sh
./dl_output/sars-cov-2_1_dcu.txt
./dl_output/sars-cov-2_2_dcu.txt
./dl_output/sars-cov-2_3_dcu.txt

./dl_output/sars-cov-2_1_cpu.txt
./dl_output/sars-cov-2_2_cpu.txt
./dl_output/sars-cov-2_3_cpu.txt
```

正确性一致说明：预期DCU版本和CPU版本的对应结果完全相同

性能说明：

a.DCU单卡比CPU单核心加速比
b.DCU单卡比CPU 32核心加速比
c.DCU多卡/单卡 加速比及并行效率
d.此算例跟成熟软件对比，正确性和性能说明
