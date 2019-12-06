#encoding=utf-8
import json

# file='result/result_webnlg_comb.txtlambda1'
def evaluate_joint(file):
	pred_all=0.0
	true_all=0.0
	pred=0.0
	for line in open(file).readlines():
		line=json.loads(line.strip())
		for item in line['content']:
			label=item['label']#pred true
			# if 1 in label:
			# 	print(item)
			if label[0]:
				p_list=[]
				for p in item['pred']:
					if 0 in p:
						break
					# else:
					# 	print(p)
					# pred_all+=1
					if p not in p_list:
						p_list.append(p)
						if p in item['true']:
							pred+=1
				pred_all+=len(p_list)
			if label[1]:
				# print(item['true'])
				true_all+=len(item['true'])
		
	print(pred, pred_all, true_all)
	p=pred/pred_all
	r=pred/true_all
	f1=2*p*r/(p+r)
	print(p,r,f1)

def evaluate_duli(file1, file2):
	pred_all=0.0
	true_all=0.0
	pred=0.0
	pred_cnn_all=0.0
	true_cnn_all=0.0
	pred_cnn=0.0
	pred_match_all=0.0
	true_match_all=0.0
	pred_match=0.0
	lines1=open(file1).readlines()
	lines2=open(file2).readlines()
	for i in range(len(lines1)):
		line1=lines1[i]
		line1=json.loads(line1.strip())
		line2=lines2[i]
		line2=json.loads(line2.strip())
		# print(line2)
		for i,item in enumerate(line1['content']):
			label=item['label']
			pred_cnn_all+=label[0]
			true_cnn_all+=label[1]
			if label==[1,1]:
				pred_cnn+=1.0
			if label==[1,1]:
				# print(type(line2['content']))
				p_list=[]
				# print(line2['content'][i])
				for p in line2['content'][i]['pred']:
					if 0 in p:
						break
					if p not in p_list:
						p_list.append(p)
						if p in line2['content'][i]['true']:
							pred+=1
				pred_all+=len(p_list)
				true_all+=len(line2['content'][i]['true'])


		
	
	
	
	if pred_cnn==0:
		p_cnn,r_cnn,f1_cnn=0,0,0
	else:
		p_cnn=pred_cnn/pred_cnn_all
		r_cnn=pred_cnn/true_cnn_all
		f1_cnn=2*p_cnn*r_cnn/(p_cnn+r_cnn)
	print(pred_cnn, pred_cnn_all, true_cnn_all)
	print(p_cnn, r_cnn, f1_cnn)
	
	true_all=1481
	print(pred, pred_all, true_all)
	p=pred/pred_all
	r=pred/true_all
	f1=2*p*r/(p+r)
	print(p,r,f1)
if __name__=="__main__":
	file='result/result_nyt_comb.txt'
	#file2='result/result_webnlg_comb.txt_duli_match'
	#file2='new_result/result_webnlg.txt'
	evaluate_joint(file)
	#evaluate_duli(file1, file2)
