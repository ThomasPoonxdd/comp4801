# from text_embedding import CLIP
# category_name_string = ';'.join(['flipflop', 'street sign', 'bracelet',
#       'necklace', 'shorts', 'floral camisole', 'orange shirt',
#       'purple dress', 'yellow tee', 'green umbrella', 'pink striped umbrella', 
#       'transparent umbrella', 'plain pink umbrella', 'blue patterned umbrella',
#       'koala', 'electric box','car', 'pole'])
# category_names = [x.strip() for x in category_name_string.split(';')]
# category_names = ['background'] + category_names
# categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
# category_indices = {cat['id']: cat for cat in categories}
# text_embedding = CLIP()
# text_feature = text_embedding(categories)
# print(text_feature)