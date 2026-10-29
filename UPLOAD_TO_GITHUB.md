# 上传到 GitHub

本目录包括完整基因表和 25 个 best checkpoints。`.gitattributes` 已将 `csv/*.csv` 和 `checkpoints/**/*.pth.tar` 配置为 Git LFS 文件。

GitHub 普通 Git 单文件上限为 100 MiB，网页上传单文件上限为 25 MiB。本目录的 BRCA、GBMLGG 基因 CSV 超过 100 MiB，部分 checkpoint 超过 25 MiB。请通过 Git 与 Git LFS 上传。

参考：[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

在 GitHub 上创建空仓库，然后在 PowerShell 中运行以下命令。将 `<YOUR_GITHUB_REPOSITORY_URL>` 替换为仓库地址：

```powershell
Set-Location 'C:\Users\admin\Desktop\SCoPSurv\SCoPSurv-Code'
git init -b main
git lfs install --local
git add .gitattributes .gitignore
git add .
git lfs ls-files
git status --short
git commit -m "Add SCoPSurv code, best checkpoints, and genomic data"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

首次 `git add` 前需要启用 Git LFS，使大文件进入 LFS 存储。上述提交会上传基因数据与权重，请确保 GitHub 账户有足够的 LFS 存储和传输额度。

克隆后获取数据与权重：

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
Set-Location <REPOSITORY_DIRECTORY>
git lfs pull
```

已包含的 `.gitignore` 只排除缓存、IDE 设置和后续训练输出；训练用 CSV、split 文件和归档 checkpoint 均会纳入版本管理。
