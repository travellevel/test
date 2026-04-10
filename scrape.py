# 爬虫练习
import requests 
from bs4 import BeautifulSoup

# 伪装主机
head={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"}

# # 这是专门练习爬虫的网站
# response=requests.get("http://books.toscrape.com/",headers=head)
# # response=requests.get("http://movie.douban.com/top250",headers=head)
# content=response.text
# soup=BeautifulSoup(content,"html.parser")

# all_price=soup.find_all("p",attrs={"class":"price_color"})
# for price in all_price:
#     print(price.string)
# # if response.ok:
# #     print(response.text)
# # else:
# #     print("失败")

import os
import undetected_chromedriver as uc
import re
"""
从www.biqubao.com笔趣阁爬取小说，楼主教程中的网址我当时没打开，
就参照楼主教程，爬取了笔趣阁小说网的内容。
    2018-07-31
"""

if __name__=='__main__':
    #所要爬取的小说主页，每次使用时，修改该网址即可，同时保证本地保存根路径存在即可
    target="https://www.biqubao.com/book/148060/"
    # 本地保存爬取的文本根路径
    save_path = r'D:\vscode\file\agent'
    #笔趣阁网站根路径
    index_path='https://www.biqubao.com'

    # 使用模拟绕过cloudflare
    driver = uc.Chrome(version_main=146)
    try:
        # 正常就使用requests就行
        driver.get(target)
        import time
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        list_tag = soup.find_all("div",attrs={"id":"list"} )
        #获取小说名称
        raw_title=list_tag[0].dl.dt.string
        match = re.search(r'《(.+?)》', raw_title)
        if match:
            story_title = match.group(1)
        # 根据小说名称创建一个文件夹,如果不存在就新建
        dir_path=save_path+'/'+story_title
        if not os.path.exists(dir_path):
            dir_path=os.path.join(save_path,story_title)
            os.mkdir(dir_path)
        #开始循环每一个章节，获取章节名称，与章节对应的网址
        for dd_tag in list_tag[0].dl.find_all('dd')[499:]:
            #章节名称
            chapter_name=dd_tag.string
            #章节网址
            chapter_url=index_path+dd_tag.a.get('href')
            #访问该章节详情网址，爬取该章节正文
            driver.get(chapter_url)
            import time
            time.sleep(5)
            html = driver.page_source
            chapter_soup = BeautifulSoup(html, "html.parser")
            #解析出来正文所在的标签
            content_tag = chapter_soup.find("div",attrs={"id":"content"})
            if content_tag is None:
                print(f"警告: {chapter_name} 未找到正文，跳过")
                continue
            #获取正文文本，并将空格替换为换行符
            content_text = content_tag.get_text(separator='\n')
            content_text = re.sub(r'\n{2,}', '\n\n', content_text)
            #将当前章节，写入以章节名字命名的txt文件
            with open(dir_path+'/'+chapter_name+'.txt', 'w',encoding='utf-8') as f:
                f.write(f'本文网址:{chapter_url}\n\n')
                f.write(content_text)
    finally:
        driver.quit()