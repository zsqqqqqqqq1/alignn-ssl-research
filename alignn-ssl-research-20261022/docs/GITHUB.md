# 发给导师与上传 GitHub

## 发给导师

直接发送本目录对应的 ZIP。请导师从根目录 README 开始，重点查看 `docs/CODE_GUIDE.md` 与 `configs/`。

ZIP 只含代码、配置、固定划分元数据和文档。不含 POSCAR、标签、缓存、模型和实验日志。需要复现时，再单独提供经校验的数据文件夹；不要把数据混入 Git 仓库。

## 新建 GitHub 仓库

建议仓库名 `alignn-ssl-research`。选择空仓库，不预先生成另一份 README。不要把整个本机科研项目上传，只使用本分享包的内容。

以下命令在解压后的项目目录执行：

```bash
git init
git add .
git status --short
git diff --cached --stat
```

确认暂存内容主要为 `ssl_alignn/`、`configs/`、`tests/`、`tools/`、`docs/`、`vendor/` 与固定划分文件，而没有训练输出，再提交：

```bash
git commit -m "Add ALIGNN masked pretraining and frozen downstream experiments"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/alignn-ssl-research.git
git push -u origin main
```

将 `YOUR_ACCOUNT` 换成自己的账号。认证由 Git/GitHub 正常处理，不要在源码、命令文档或远程 URL 中保存访问令牌。本次整理没有替你创建远程仓库或推送。

也可以通过 GitHub 网页上传解压后的文件夹内容；不要只上传 ZIP，否则浏览者难以直接查看代码、差异和 README。

## 已配置的忽略规则

`.gitignore` 排除数据内容、模型、缓存、日志、运行目录、Python 缓存和本地环境文件。它不会移除已被追踪的文件，因此每次提交前仍查看暂存列表。

`.gitattributes` 为 shell 保留 LF 换行，并保持参与实验身份验证的配置、固定划分及上游文件的原始字节，减少跨平台换行改写造成的哈希差异。

## 许可证与署名

`vendor/alignn/LICENSE.rst` 是随上游文件保留的 NIST 原始条款。本项目新增研究代码还没有指定开放许可证，根目录 `LICENSE.md` 已明确区分。不要直接将整个仓库标注为 MIT，或删除上游许可声明。

如果后续决定正式开放源代码，再由作者/课题组明确选定新增代码的许可证，并更新 LICENSE 与仓库设置。代码可见与已授予开源许可是两个不同事项。

## 后续更新

更新训练实现时，同步修改方法文档及测试，并为实验保留独立输出目录。不要为了通过断点校验而手动篡改 `protocol.json`、源码指纹或断点签名。
