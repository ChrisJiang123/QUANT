from html.parser import HTMLParser
from v4_bootstrap import ROOT
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if any(x in href.lower() for x in ['.zip','funding']):print(href)
if __name__=='__main__':Links().feed((ROOT/'experiments/v4/public/kraken_funding_export_docs.raw').read_text(encoding='utf-8'))
