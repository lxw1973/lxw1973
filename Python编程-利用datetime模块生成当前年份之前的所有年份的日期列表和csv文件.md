# Python编程-利用datetime模块生成当前年份之前的所有年份的日期列表和csv文件

-----

今天学习Pandas日期数据处理，需要利用年月日的日期数据而且是csv文件格式。为了数据准确严谨，于是编写了这个程序，感兴趣的可以根据自己需要进行修改。分享源码如下：

```python
#_*_coding:utf-8_*_
# 作者      ：liuxiaowei
# 创建时间   ：3/24/22 10:56 AM
# 文件      ：生成年月日csv.py
# IDE      ：PyCharm

import csv
import datetime

def create_ymd_lst():
    import datetime

    nd = datetime.date.today()
    date_lst = []

    y = int(input('输入数字：'))
    for i in range(y):
        year = nd.year - i
        date_lst.append(str(year))
    mon_day_lst = []
    for m in range(1, 13):
        if m == 2:
            for d in range(1, 29):
                if d < 10:
                    mon_day_lst.append('-02-0' + str(d))
                else:
                    mon_day_lst.append(('-02-') + str(d))

        elif m in [4, 6, 8, 9, 11]:
            for d in range(1, 31):
                if d < 10 and m != 11:
                    mon_day_lst.append(str(f'-0{m}-0' + str(d)))
                elif d >= 10 and m != 11:
                    mon_day_lst.append(str(f'-0{m}-' + str(d)))
                elif d < 10:
                    mon_day_lst.append(str(f'-{m}-0' + str(d)))
                else:
                    mon_day_lst.append(str(f'-{m}-' + str(d)))
        else:
            for d in range(1, 32):
                if d < 10 and m < 10:
                    mon_day_lst.append(str(f'-0{m}-0' + str(d)))
                elif d >= 10 and m < 10:
                    mon_day_lst.append(str(f'-0{m}-' + str(d)))
                elif d < 10 and m >= 10:
                    mon_day_lst.append(str(f'-{m}-0' + str(d)))
                else:
                    mon_day_lst.append(str(f'-{m}-') + str(d))
    # 对年份按生序排序
    new_year_lst = list(reversed(date_lst))
    # 年月日空列表，然后添加年月日字符串
    ymd_lst = []
    for year in new_year_lst:
        for md in mon_day_lst:
            ymd_lst.append(year + md)
            print(year + md)
	# 返回一个年月日列表
    return ymd_lst

 # 调用 生成年月日列表的函数
ymd_lst = create_ymd_lst()
# 数字序号列表，从3开始
num_lst = []

for num in range(len(ymd_lst)):
    num_lst.append(num)



date_dict = {'date':ymd_lst, 'number': num_lst}
with open('date.csv', 'w') as f:
    writer = csv.writer(f)

    writer.writerow(date_dict)
    # 循环的次数是根据字典的值的长度设定，取决于添加多少个元素
    for i in range(len(ymd_lst)):
            # 每循环一次生成一个临时列表，然后写入csv文件
        templist = []
        templist.append(ymd_lst[i])
        templist.append(num_lst[i]+3)
            # 写对象把每行数据写入csv
        writer.writerow(templist)

```

运行结果如下：

部分数据

![](https://tva1.sinaimg.cn/large/e6c9d24ely1h0kujfcksij20je0q8q4v.jpg)