# vim

## 基本命令表格
鸟哥的linux私房菜第四版　p461

## 区块选择
p471

## 多文件编辑
通过`vim file1 file2`打开两个文件，并且在其中一个文件拷贝到粘贴板上的文本可以在另一个文件中使用。
p473

## 多窗口功能
p474

`:sp {filename}`
- 加了文件名，则是多窗口不同文件
- 不加文件名，直接sp出现的是同一个文件在不同窗口

## 补全功能
- `[ctrl]+n`:
- `[ctrl]+f`:
- `[ctrl]+o`:

## vim 环境参数设置
在文件`/etc/vimrc`与文件`~/.vimrc`中进行设置。其中默认第二个不存在，需要自己手动创建。

p478


## 其他注意事项
### 编码问题
- linux系统默认支持的语系数据：这与`/etc/locale.conf`有关
- 终端接口(bash)的语系：与`LANG`、`LC_ALL`几个环境变量有关
- 文件的编码方式
- 打开文件的软件的接口的编码方式

### windows与linux的文件换行符的差别

### 语系编码转换

