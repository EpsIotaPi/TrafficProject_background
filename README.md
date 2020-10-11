# README

this is a project for HDU Traffic Project, and only use for study

this is the background part

### Virtual Environment config
```
python3.7
1. Flask
2. flask_cors
3. PuLP-py3
4. pandas
5. xlrd

for real time predict:
1. tensorflow version 1.15
```
you can execute following command in shell to create virtual environment with conda.
```
conda create -n py3.7_traffic python=3.7
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install Flask flask_cors PuLP-py3 pandas xlrd
pip install tensorflow==1.15
```

##### PS:
* home page's Time is Cal by now to last Monday's 8 o'clock
* we have four incident type:
    * 「占用车道」「分流、限流」「借道通行」「其它」
* the incident_status is generate follow by: 
    * 45% 「进行中」
    * 30% 「已完成」
    * 20% 「等待中」
    * 5%  「异常」
* We follow the principle of rounding down when dealing with float
* the output data is always with 「调用成功」 message