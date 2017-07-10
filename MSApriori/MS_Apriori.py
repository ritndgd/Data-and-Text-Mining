import operator
import itertools

def main():
    MIS_dict, sdc, cbt, mh, n, T = data_read()
    MSApriori(MIS_dict, sdc, cbt, mh, n, T)

def MSApriori(MIS_dict, sdc, cbt, mh, n, T):
    M_temp = sorted(MIS_dict.items(), key=operator.itemgetter(1))
    L_count={}
    M=[]
    for i in range(0, len(M_temp)):
        M.append(M_temp[i][0])
    L = init_pass(M, n, MIS_dict, T)
    F1=[]
    k=2
    for i in range(0,len(L)):
        cnt=0
        for ele in T:
            cnt=cnt+ele.count(L[i])
        L_count[L[i]]=cnt;
        temp_list=[]
        if((cnt/n)>=MIS_dict[L[i]]):
            temp_list.append(L[i])
            F1.append(temp_list)
    F1=must_have(F1,mh)
    cnt=len(F1)
    print("Frequent 1-itemsets")
    for item in F1:
        print("\t",L_count[item[0]],":", item)
    print("Total number of frequent 1-itemsets =",len(F1))
    while(len(F1)>0):
        C=[]
        F=[]
        count_dict={}
        tail_c_dict={}
        if(k==2):
            C=level2_candidate_gen(L,sdc,L_count,n,MIS_dict)
        else:
            
            C=MScandidate_gen(F1,sdc,k,MIS_dict,L_count,n)

        for t in T:
            t_set=set(t)
            for c in C:
                c_set=set(c)
                if c_set.issubset(t_set):
                    if repr(c_set) in count_dict:
                        count_dict[repr(c_set)]=count_dict[repr(c_set)]+1
                    else:
                        count_dict[repr(c_set)]=1
                c2=[]
                for i in range(1,len(c)):
                    c2.append(c[i])
                c2_set=set(c2)
                if c2_set.issubset(t_set):
                    if repr(c_set) in tail_c_dict:
                        tail_c_dict[repr(c_set)]=tail_c_dict[repr(c_set)]+1
                    else:
                        tail_c_dict[repr(c_set)]=1
        for c in C:
            if(float(count_dict[repr(set(c))]/n) >= MIS_dict[c[0]]):
                F.append(c)
        if(len(F)>0):
            F=must_have(F,mh)
            F=cannotBtogthr(F,cbt)
            print("Frequent",k,"-itemsets")
            for item in F:
                print("\t",count_dict[repr(set(item))],":",item)
                print("Tailcount:",tail_c_dict[repr(set(item))])
            print("Total number of frequent",str(k)+"-itemsets: ", len(F))
            print()
        k=k+1
        F1=F

def level2_candidate_gen(L,sdc,L_count,n,MIS_dict):
    C2=[]
    for i in range(0,len(L)):
        sup_l=float(L_count[L[i]]/n)
        if (sup_l >= MIS_dict[L[i]]):
            for j in range(i+1, len(L)):
                sup_h= float(L_count[L[j]]/n)
                if((sup_h >= MIS_dict[L[i]]) and (abs(sup_h-sup_l)<=sdc)):
                    C2.append([L[i],L[j]])
    return C2

def MScandidate_gen(F, sdc, k, MIS_dict, L_count, n):
    C=[]
    f1=[]
    f2=[]
    if(len(F)>1):
        for i in range(0,len(F)):
            f1=F[i]
            for m in range(i+1,len(F)):
                c=[]
                f2=F[m]
                has_same=True
                for j in range(0,len(f1)-1):
                    if(f1[j]!=f2[j]):
                        has_same=False
                        break
                if(has_same==True):
                    sup_lf1=float(L_count[f1[len(f1)-1]]/n)
                    sup_lf2=float(L_count[f2[len(f2)-1]]/n)
                    if (f1[len(f1)-1]!=f2[len(f2)-1] and abs(sup_lf1-sup_lf2)<=sdc ):
                        c=f1[:]
                        c.append(f2[len(f2)-1])
                if(len(c)>0):
                    C.append(c)
                    c_set=itertools.combinations(c,k-1)
                    for ss in set(c_set):
                        if((set(c[0]).issubset(ss)) or (MIS_dict[c[1]]==MIS_dict[c[0]])):
                            if(ss.issubset(set(F))==False):
                                C.remove(c)    
    return C        
    

def must_have(F_set,mh):
    F=[]
    for item in F_set:
        boo=False
        for ele in mh:
            if(ele in item):
                boo=True
        if(boo==True):
            F.append(item)
    return F

def cannotBtogthr(F_set, cbt):
    F_set_new=[]
    for alist in cbt:
        for aset in F_set:
            if not set(alist).issubset(set(aset)):
                F_set_new.append(aset)        
    return F_set

def init_pass(M,n, MIS_dict, T):
    L=[]
    minsup_f=MIS_dict[M[0]]
    for i in range(0, len(M)):
        f_item=M[i]
        cnt=0
        for ele in T:
            cnt=cnt+ele.count(f_item)
        sup = float(cnt/n)        
        if(sup>=minsup_f):
            L.append(f_item)
    return L
    
def data_read():
    pmtr_file=open("parameter-file.txt")
    inp_data=open("input-data.txt")
    MIS_dict={}
    sdc=None
    cbt=[]
    mh=[]
    n=None
    inp_data_list=[]
    for line in pmtr_file.readlines():
        if(line.split("(")[0]=="MIS"):
            k=float((line.split("=")[1]).strip())
            j=((line.split("(")[1]).split(")")[0])
            MIS_dict[j]=k
        else:
            if((line.split("=")[0]).strip()=="SDC"):
                sdc=float((line.split("=")[1]).strip())
            elif((line.split(":")[0]).strip()=="cannot_be_together"):
                temp=(line.split(":")[1]).strip()
                temp=temp+","
                for ele in temp.split("},"):
                    ele=ele.strip()
                    ele=ele[1:len(ele)]
                    kk=[]
                    if(len(ele)>0):
                        for subele in ele.split(","):
                            kk.append(subele.strip())
                        cbt.append(kk)
            else:
                temp=(line.split(":")[1]).strip()
                for ele in temp.split("or"):
                    mh.append(ele.strip())

    inp_lines=inp_data.readlines()
    n=len(inp_lines)
    for line in inp_lines:    
        line=line.split("{")[1]
        line=line.split("}")[0]
        temp=[]
        for ele in line.split(","):
            temp.append(ele.strip())
        inp_data_list.append(temp)
    
    return MIS_dict,sdc,cbt,mh,n,inp_data_list



if __name__ == '__main__':
    main()
