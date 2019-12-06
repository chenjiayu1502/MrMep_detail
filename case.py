#encode=utf-8
import json

def reverse_dict(dic):
	new_dic={}
	for k,v in dic.items():
		new_dic[v]=k
	return new_dic


# data_dir='data/ske_token/'
data_dir='data/nyt_seq/'
result_file='result/result_nyt_comb.txt'
f=open('case_result_for_nyt.txt','w')



# data_dir='data/webnlg_seq/'
# result_file='result/result_webnlg_comb.txt'
# f=open('case_result_for_webnlg.txt','w')



sen_file=data_dir+'dev.ids.context'
vocab_file=data_dir+'words2id.json'
relation_file=data_dir+'relations2id.json'
match_file=data_dir+'dev_match_output.json'
w2id=json.load(open(vocab_file,encoding='utf-8'))
id2w=reverse_dict(w2id)
r2id=json.load(open(relation_file,encoding='utf-8'))
id2r=reverse_dict(r2id)


sentence=[]
for line in open(sen_file).readlines():
	line=line.strip().split()
	sen=[id2w[int(k)] for k in line[1:]]
	sentence.append(sen)
print(sentence[0])

gold_data=[]
for ind,line in enumerate(open(match_file).readlines()):
	line=json.loads(line.strip())
	temp=[]
	for i,item in enumerate(line['data']):
		if item==[]:
			continue
		rel=id2r[i]
		# print(rel)
		# print(sentence[ind])
		for [s1,e1,s2,e2] in item:
			# print(s1,e1,s2,e2)
			en1=sentence[ind][s1-1:e1]
			en2=sentence[ind][s2-1:e2]
			# print(en1,en2)
			temp.append((' '.join(en1),rel,' '.join(en2)))
			# break
		# break
	gold_data.append(temp)
print(gold_data[0])

pred_data=[]
for ind,line in enumerate(open(result_file).readlines()):
	line=json.loads(line.strip())
	temp=[]
	for item in line['content']:
		# print(item)
		rel=id2r[item['rel']]
		if 0 in [s1,e1,s2,e2]:
			continue
		for [s1,e1,s2,e2] in item['pred']:
			en1=sentence[ind][s1-1:e1]
			en2=sentence[ind][s2-1:e2]
			temp.append((' '.join(en1),rel,' '.join(en2)))
	pred_data.append(temp)
	
f=open('case_result_for_nyt.txt','w',encoding='utf-8')
all_data={"data":[]}
for i in range(len(gold_data)):
	g_rel=[r[1] for r in gold_data[i]]
	p_rel=[r[1] for r in pred_data[i]]
	sorted(g_rel)
	sorted(p_rel)
	# if gold_data[i]!=pred_data[i] and len(gold_data[i])>3:
	sen=''.join(sentence[i])
	
	if 'UNK' not in sen and len(gold_data[i])>1:
		pdata=''
		print(gold_data[i])
		for p in gold_data[i]:
			print(p)
			pdata+=str(p)+"$"
		#print(pdata)
		pdata=pdata[:-1]
		#print(pdata)
		item_data={'sen':' '.join(sentence[i]),
					'gold':gold_data[i],
					'pred':pdata}
		all_data["data"].append(item_data)
		if len(all_data["data"])>100:
			break
f.write(json.dumps(all_data,ensure_ascii=False,indent=4)+'\n')


# print(len(gold_data))
# print(len(pred_data))
# all_data={"data":[]}
# for i in range(len(gold_data)):
# 	g_rel=[r[1] for r in gold_data[i]]
# 	p_rel=[r[1] for r in pred_data[i]]
# 	sorted(g_rel)
# 	sorted(p_rel)
# 	# if gold_data[i]!=pred_data[i] and len(gold_data[i])>3:
# 	sen=' '.join(sentence[i])
# 	#print(sen)
	
# 	if 'UNK' not in sen and len(gold_data[i])>1:
		
# 		item_data={'sen':sen,
# 					'gold':str(gold_data[i]),
# 					'pred':pred_data[i]}
# 		all_data["data"].append(item_data)
# 		if len(all_data["data"])>10:
# 			break
# 		f.write(json.dumps(item_data,ensure_ascii=False,indent=4)+'\n')

# 		#f.write(str(i)+'\n')
# 		#f.write(''.join(sentence[i])+'\n')
# 		#f.write(str(gold_data[i])+'\n')
# 		#f.write(str(pred_data[i])+'\n')
# 		# f.write(len(gold_data[i]))
# 	