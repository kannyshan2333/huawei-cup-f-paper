# 华为杯 F 题论文协作仓库

本仓库从 `E:\华为杯f\华为杯论文_Overleaf上传包.zip` 整理而来，保留论文全部 `.tex` 源文件、`format.cls`、字体、封面图、正文图片和附录引用的四份 Python 代码。原包中的大型结果 CSV、notebook 和原始数据未纳入协作仓库；它们不参与论文编译。

## 编辑与编译

- 主文件：`论文.tex`。
- 编译器：**XeLaTeX**；本地可执行 `latexmk -xelatex -interaction=nonstopmode -halt-on-error 论文.tex`。
- 在 GitHub 修改后，**Actions → 编译论文** 会自动编译；成功后在该次运行的 **Artifacts** 下载 `论文-PDF`。
- 也可把仓库根目录中的 `.tex`、`format.cls`、`fonts/`、`figures/`、`求解/` 上传到 Overleaf。将主文件设为 `论文.tex`，编译器设为 XeLaTeX。

## 协作建议

请各自修改不同的章节文件；较大的改动在分支中提交 Pull Request，编译通过后合并。不要把 `real_attachments` 原始数据或本地编译产生的辅助文件上传到仓库。论文中的数值结论以已有稿件为基线，尚未重新计算。

GitHub 与 Overleaf 的直接同步需 Overleaf 的付费功能，并且只能用**新建**的 Overleaf 项目连接现有 GitHub 仓库；原有 Overleaf 项目不能直接绑定这个已有仓库。没有该功能时，可在 GitHub 协作并通过 Actions 获取编译 PDF，或手动将源码导入 Overleaf。
