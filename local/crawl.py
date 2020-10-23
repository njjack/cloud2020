import urllib.request as req
import json
import bs4

def getdata(url, target, rtwa, rtitle, rurl):
    request = req.Request(url, headers={
        "cookie":"over18=1",
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
    })

    with req.urlopen(request) as response:
        raw = response.read().decode("utf-8")

    page = bs4.BeautifulSoup(raw, "html.parser")
    items = page.find_all("div", class_="title", )
    count = 0
    for item in items:
        if item.a != None:
            item2 = item.find_previous_sibling('div')
            if item2.span == None:
                twa = "0"
            else:
                twa = item2.span.string
            if twa[0] == "X":
                continue
            if twa[0] =="爆":
                twa = "100"
            if int(twa) >= target:
                count = count + 1
                rtwa.append(twa)
                rurl.append(item.a.get("href"))
                rtitle.append(item.a.string)
    next = page.find("a", string = "‹ 上頁")
    return count, "https://www.ptt.cc"+next["href"]

def crawl(target):
    PTT_URL = "https://www.ptt.cc/bbs/Gossiping"
    count = 0
    rtwa = []
    rtitle = []
    rurl = []
    while(count < 3):
        tmp, PTT_URL = getdata(PTT_URL, target, rtwa, rtitle, rurl)
        count += tmp
    return rtwa, rtitle, rurl