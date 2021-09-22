import json
import numpy as np

def get_compats(polyvore_split='nondisjoint'):
    dataset_path = 'data/polyvore_outfits/{}/'.format(polyvore_split)
    
    json_file = dataset_path + '{}.json'.format('test')
    with open(json_file) as f:
        json_data = json.load(f)

    map_ids = {}
    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            map_ids[outfit_id] = item['item_id']


    compat_file = dataset_path + 'compatibility_test.txt'
    compat_file = dataset_path + 'fashion_compatibility_harder_test.txt'
    outfits = []
    n_comps = 0
    
    with open(compat_file) as f:
        for line in f:
            cols = line.rstrip().split(' ')
            compat_score = float(cols[0])
            assert compat_score in [1, 0]
            
            try:
                map_ids[cols[1]]
                items=cols[1:]
            except:
                items = cols[2:]
            
            # map their ids to my img ids
            items = [map_ids[it] for it in items]
            n_comps += 1
            outfits.append((items, compat_score))

    print('There are {} outfits to test compatibility'.format(n_comps))
    print('There are {} test outfits.'.format(len(json_data)))

    return outfits

def get_questions(polyvore_split='nondisjoint'):
    dataset_path = 'data/polyvore_outfits/{}/'.format(polyvore_split)
    questions_file = dataset_path + 'fill_in_blank_test.json'    
    questions_file = dataset_path + 'fill_in_blank_test_harder_new.json'
    json_file = dataset_path+"test.json"
    
    with open(json_file) as f:
        json_data = json.load(f)
        
    with open(questions_file) as f:
        questions_data = json.load(f)
    
    print('There are {} questions.'.format(len(questions_data)))
    print('There are {} test outfits.'.format(len(json_data)))
    
    map_ids = {}
    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            map_ids[outfit_id] = item['item_id']
    
    
    save_data = []

    for ques in questions_data:
        """ques: answers, blank_position, question"""
        q = []
    
        for q_id in ques['question']:
            outfit_id = q_id.split('_')[0]#set_id
            q_id = map_ids[q_id]#item_id
            q.append(q_id)
        a = []
        positions = []
        i = 0
        for a_id in ques['answers']:
            if i == 0:
                assert a_id.split('_')[0] == outfit_id
            else:
                if a_id.split('_')[0] == outfit_id:
                    pass # this is true for a few edge queries
            pos = int(a_id.split('_')[-1]) # get the position of this item within the outfit
            a_id = map_ids[a_id]#answer set_id to item_id
            a.append(a_id)
            positions.append(pos)#position of set_id for each question
            i += 1
        """q: question item_id; 
           a: answer item_id"""
        save_data.append([q, a, positions, ques['blank_position']])
    return save_data

if __name__=="__main__":
    outfits = get_compats()#(20000)
    questions = get_questions()#(10000)
    """
    questions[i]:
            list: question item_ids
            list: answer item_ids
            list: indexes(categories) of answer set_id
            int: blank(right) index(category) of the right answer set_id
        
    outfits[i]:
            list: item_ids
            float: compat_score, 0/1
    """










