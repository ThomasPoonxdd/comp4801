import os
import pandas as pd
from snscrape.modules import twitter
import json
import requests
from bs4 import BeautifulSoup

class TwitterScraper:
    def __init__(self):
        self.output_folder = './data'
        self.hashtags = ['LostAndFound']
        self.usernames = ['LostAndFoundLON']
        self.max_results = 1000

    def create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def scrape_hashtag(self, hashtag):
        scraper = twitter.TwitterHashtagScraper(hashtag)
        return scraper
    
    def scrape_user(self, user):
        scraper = twitter.TwitterUserScraper(user)
        return scraper
    
    def start_scraping_hashtag(self):
        output_folder_hashtag = self.output_folder + '/hashtags'
        if not os.path.exists(output_folder_hashtag):
            os.makedirs(output_folder_hashtag)
        for hashtag in self.hashtags:
            output_filename = output_folder_hashtag + "/" + hashtag.replace(" ", "_") + ".csv"
            if not os.path.exists(output_filename):
                df = pd.DataFrame(columns=["id", "text", "timestamp", "replyCount", "retweetCount", "likeCount", "quoteCount"])
            else:
                df = pd.read_csv(output_filename)

            scraper = self.scrape_hashtag(hashtag)

            i = 0
            for i, tweet in enumerate(scraper.get_items(), start = 1):
                tweet_json = json.loads(tweet.json())

                id = str(tweet_json['id'])
                text = '"' + str(tweet_json['rawContent']).replace("\n", "") + '"'
                timestamp = str(tweet_json['date'])
                viewCount = str(tweet_json['viewCount'])
                replyCount = str(tweet_json['replyCount'])
                retweetCount = str(tweet_json['retweetCount'])
                likeCount = str(tweet_json['likeCount'])
                quoteCount = str(tweet_json['quoteCount'])

                data = {"id": id, 
                        "text": text, 
                        "timestamp": timestamp,
                        "viewCount": viewCount, 
                        "replyCount": replyCount, 
                        "retweetCount": retweetCount,  
                        "likeCount": likeCount, 
                        "quoteCount": quoteCount
                        }
                    
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

                if self.max_results and i > self.max_results:
                    break
            
            df = df.drop_duplicates(keep=False)
            df.to_csv(output_filename, encoding='utf-8', index=False)

    def start_scraping_user(self):
        output_folder_hashtag = self.output_folder + '/usernames'
        if not os.path.exists(output_folder_hashtag):
            os.makedirs(output_folder_hashtag)
        for user in self.usernames:
            output_filename = output_folder_hashtag + "/" + user.replace(" ", "_") + ".csv"
            if not os.path.exists(output_filename):
                df = pd.DataFrame(columns=["id", "text", "timestamp", "replyCount", "retweetCount", "likeCount", "quoteCount"])
            else:
                df = pd.read_csv(output_filename)

            scraper = self.scrape_user(user)

            i = 0
            for i, tweet in enumerate(scraper.get_items(), start = 1):
                tweet_json = json.loads(tweet.json())


                id = str(tweet_json['id'])
                text = ('"' + str(tweet_json['rawContent']).replace("\n", "") + '"').split()
                timestamp = str(tweet_json['date'])
                viewCount = str(tweet_json['viewCount'])
                replyCount = str(tweet_json['replyCount'])
                retweetCount = str(tweet_json['retweetCount'])
                likeCount = str(tweet_json['likeCount'])
                quoteCount = str(tweet_json['quoteCount'])

                try:
                    for t in text:
                        if t.startswith("https:"):
                            url = t
                            break
                    r = requests.get(url, timeout=5)
                    soup = BeautifulSoup(r.text, "html.parser")
                    text = soup.find('div', id="FeaturedIncident").find('p', class_="incident-text")
                    text = text.text

                    data = {"id": id, 
                        "text": text, 
                        "timestamp": timestamp, 
                        "viewCount": viewCount,
                        "replyCount": replyCount, 
                        "retweetCount": retweetCount,  
                        "likeCount": likeCount, 
                        "quoteCount": quoteCount
                        }
                    
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                
                except KeyboardInterrupt:
                    exit()
                except:
                    pass

                if self.max_results and i > self.max_results:
                    break
            
            df = df.drop_duplicates(keep=False)
            df.to_csv(output_filename, encoding='utf-8', index=False)

    def start(self):
        self.create_output_folder()
        # self.start_scraping_hashtag()
        self.start_scraping_user()

scraper = TwitterScraper()
scraper.start()


    


    

# output_folder = './data'
# hashtags = ['LostAndFound']
# usernames = ['LostAndFoundLON']
# max_results = 100

# def create_output_folder(output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

# def scrape_hashtag(hashtag):
#     scraper = twitter.TwitterHashtagScraper(hashtag)
#     return scraper

# def scrape_user(user):
#     scraper = twitter.TwitterUserScraper(user)
#     return scraper

# def start_scraping_hashtag(hashtags):
#     output_folder_hashtag = output_folder + '/hashtags'
#     if not os.path.exists(output_folder_hashtag):
#         os.makedirs(output_folder_hashtag)
#     for hashtag in hashtags:
#         output_filename = output_folder_hashtag + "/" + hashtag.replace(" ", "_") + ".csv"
#         if not os.path.exists(output_filename):
#             df = pd.DataFrame(columns=["id", "text", "timestamp", "replyCount", "retweetCount", "likeCount", "quoteCount"])
#         else:
#             df = pd.read_csv(output_filename)

#         scraper = scrape_hashtag(hashtag)

#         i = 0
#         for i, tweet in enumerate(scraper.get_items(), start = 1):
#             tweet_json = json.loads(tweet.json())

#             id = str(tweet_json['id'])
#             text = '"' + str(tweet_json['rawContent']).replace("\n", "") + '"'
#             timestamp = str(tweet_json['date'])
#             replyCount = str(tweet_json['replyCount'])
#             retweetCount = str(tweet_json['retweetCount'])
#             likeCount = str(tweet_json['likeCount'])
#             quoteCount = str(tweet_json['quoteCount'])

#             data = {"id": id, 
#                     "text": text, 
#                     "timestamp": timestamp, 
#                     "replyCount": replyCount, 
#                     "retweetCount": retweetCount,  
#                     "likeCount": likeCount, 
#                     "quoteCount": quoteCount
#                     }
                
#             df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

#             if max_results and i > max_results:
#                 break
        
#         df = df.drop_duplicates(keep=False)
#         df.to_csv(output_filename, encoding='utf-8', index=False)

# def start_scraping_user(usernames):
#     output_folder_hashtag = output_folder + '/usernames'
#     if not os.path.exists(output_folder_hashtag):
#         os.makedirs(output_folder_hashtag)
#     for hashtag in hashtags:
#         output_filename = output_folder_hashtag + "/" + hashtag.replace(" ", "_") + ".csv"
#         if not os.path.exists(output_filename):
#             df = pd.DataFrame(columns=["id", "text", "timestamp", "replyCount", "retweetCount", "likeCount", "quoteCount"])
#         else:
#             df = pd.read_csv(output_filename)

#         scraper = scrape_user(hashtag)

#         i = 0
#         for i, tweet in enumerate(scraper.get_items(), start = 1):
#             tweet_json = json.loads(tweet.json())

#             id = str(tweet_json['id'])
#             text = '"' + str(tweet_json['rawContent']).replace("\n", "") + '"'
#             timestamp = str(tweet_json['date'])
#             replyCount = str(tweet_json['replyCount'])
#             retweetCount = str(tweet_json['retweetCount'])
#             likeCount = str(tweet_json['likeCount'])
#             quoteCount = str(tweet_json['quoteCount'])

#             data = {"id": id, 
#                     "text": text, 
#                     "timestamp": timestamp, 
#                     "replyCount": replyCount, 
#                     "retweetCount": retweetCount,  
#                     "likeCount": likeCount, 
#                     "quoteCount": quoteCount
#                     }
                
#             df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

#             if max_results and i > max_results:
#                 break
        
#         df = df.drop_duplicates(keep=False)
#         df.to_csv(output_filename, encoding='utf-8', index=False)

# def main():
#     create_output_folder(output_folder)
#     start_scraping_hashtag(hashtags)
#     start_scraping_user(usernames)

