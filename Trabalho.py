import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import datetime
import random
import string

#Dados do problema
area = 5850000 #[m²]
#L = 3.665 #comprimento [km]
maxF1 = 1 #Valor máximo da função objetivo 1
maxF2 = 1 #Valor máximo da função objetivo 2
minF1 = 0 #Valor mínimo da função objetivo 1
minF2 = 0 #Valor mínimo da função objetivo 2
pop_i = 40 # Dimensão da população
Pr = 0.9 #Probabilidade de recombinação
Pm = 0.5 #Probabilidade de mutação
GER = 50 #Número máximo de gerações

#Lendo os dados
prec = pd.read_csv('prec.csv',sep=';', encoding = "ISO-8859-1") #Q [m³/s]
hid_obs = pd.read_csv('hid_obs.csv',sep=';', encoding = "ISO-8859-1") #P [mm/min]

#Renomeando colunas
hid_obs.rename(columns={'Data e Hora':'date','Q(m3/s)':'flow'},inplace=True)
prec.rename(columns={'Data e Hora':'date','Pmédia (mm/min)':'prec'},inplace=True)

#Transformando os valores de data para o formato datetime
hid_obs['date']=pd.to_datetime(hid_obs['date'], infer_datetime_format=True)
prec['date']=pd.to_datetime(prec['date'], infer_datetime_format=True)
ptotal = prec['prec'].sum()

"""#Plotando o hidrograma para análise
ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis_date()
plt.plot(hid_obs['date'],hid_obs['flow'])
plt.show()"""

peak1 = pd.to_datetime('25/02/2015 19:24',infer_datetime_format=True)
peak2 = pd.to_datetime('25/02/2015 22:46',infer_datetime_format=True) 
hid_between_peaks = hid_obs.ix[hid_obs.index[hid_obs['date'] == peak1].tolist()[0]:hid_obs.index[hid_obs['date'] == peak2].tolist()[0]]
qcut = hid_between_peaks['flow'].min()
d = hid_between_peaks[hid_between_peaks['flow'] == qcut]

#Criando novo hidrograma com apenas um pico
hidwbase = hid_obs.ix[0:d.index.values[0]]
zeros = hidwbase[hidwbase['flow'] == 0]
last_zero = zeros[zeros['date'] == zeros['date'].max()]
new_hid = hidwbase[hidwbase['flow'] != 0]

#Tempo entre a vazão mínima entre picos e o começo do hidrograma
time_dif =  d.iloc[0]['date'] - last_zero.iloc[0]['date'] 

#Calculando o coeficiente angular da reta de separação do escoamento de base
m = (d.iloc[0]['flow']-last_zero.iloc[0]['flow'])/int(round(time_dif.total_seconds() / 60))

#Retirando o escoamento de base do hidrograma
separation_line = []
for i, row in new_hid.iterrows():
    td = row['date'] - d.iloc[0]['date']
    t_min = int(round(td.total_seconds() / 60))
    value = m*(t_min) + qcut
    separation_line.append(value)
    
new_hid['sep_line']=separation_line
new_hid['wo_base']=new_hid['flow']-new_hid['sep_line']

media = new_hid['flow'].mean() #vazão média observada
max_obs = new_hid['flow'].max() #vazão máxima observada


#Calculando o intervalo de discretização
tc_timestamp = new_hid.iloc[117]['date'] - prec.iloc[160]['date'] 
tc = tc_timestamp.total_seconds()/60 #tempo de concentração em minutos
disc_interval = int(tc/5)

#Calculando a duração total do escoamento em minutos
r_duration_timestamp = new_hid.iloc[117]['date'] - new_hid.iloc[0]['date']
r_duration = r_duration_timestamp.total_seconds()/60

#Calculando o número de variáveis de decisão
variables = int(r_duration / disc_interval)+1
               

time_intervals = []
a=0
for i, row in prec.iterrows():
    previous_row = i - 1
    if previous_row < 0:
        deltatime_seconds=0
    else:
        timestamp = row['date']-prec.iloc[previous_row]['date']
        deltatime_seconds = timestamp.total_seconds()
    a = a + deltatime_seconds
    time_intervals.append(a)
prec['Nd']=time_intervals
       
total_prec = prec['prec'].sum()

infil = []


infiltration = pd.DataFrame({'q_infil':np.full(9, np.nan).tolist()})
qinfiltration = pd.concat([infiltration,new_hid['sep_line']], ignore_index=True)
prec = pd.concat([prec,qinfiltration[0]],axis=1)

prec = prec.rename(columns={0: 'qinf'})
prec['pinf']=prec['qinf']*60000/area
prec.pinf.fillna(prec.prec, inplace=True)   
prec['Pef']=prec['prec']-prec['pinf']

prec = prec[prec['prec']!=0]

#Calculando a duração total da precipitação em minutos
p_duration_timestamp = prec.iloc[55]['date'] - prec.iloc[0]['date'] 
p_duration = p_duration_timestamp.total_seconds()/60

def create_pop(pop_size,num_variables):
    var_names = []
    aux = 1
    for i in range(pop_size):
        var_names.append('CN'+str(aux))
        aux = aux +1
    CN_sample = np.random.uniform(50,100,pop_size)
    aux = 0
    CN = {}
    for i in CN_sample:
        CN[var_names[aux]]=[i]
        aux=aux+1
        CN_pop = pd.DataFrame(CN,columns=var_names)
    aux = 0
    cotas = {}
    for i in range(pop_size):
        cotas[var_names[aux]]=np.random.uniform(0,new_hid['flow'].max(),num_variables)
        aux=aux+1
        Y = pd.DataFrame(cotas,columns=var_names)
        input_data=pd.concat([CN_pop,Y],ignore_index=True)
    return (input_data)

def prec_efetiva(P,CN,p_duration,disc_interval):
    S = 25.4*(1000/CN-10)
    A = 0.2*S
    l = []
    time = P.iloc[0]['Nd']
    au = 0
    while au < p_duration:
        l.append(time)
        au = au + disc_interval
        time = time + disc_interval*60
    p_discrete =[]
    aux = 0
    for i in l:
        if aux == 0:
            ant = 0
        else:
            ant = l[aux-1]
        p = prec[prec['Nd'].between(ant, i, inclusive=True)]
        p_discrete.append(p['prec'].sum())
        aux = aux + 1
    Pef = []
    for j in p_discrete:
        if A<j:
            Pe=((j-A)**2)/(j-A+S)
        else:
            Pe=0
        Pef.append(Pe)
    return Pef

def convolution(pop,Pef):
    num_sol = len(pop.columns)
    hydrovar_names = []
    hydrographs = {}
    aux = 0
    CN = pop.iloc[:1]
    pop_wo_CN = pop.iloc[1:]
    for column in pop_wo_CN:
        HU = pop_wo_CN[column]
        P = Pef[aux]
        hydrograph =  np.convolve(HU,P).tolist()
        aux = aux+1
        key = 'hyd'+'CN'+str(aux)
        hydrovar_names.append(key)
        hydrographs[key]=hydrograph
    hydrographs = pd.DataFrame(hydrographs,columns=hydrovar_names)
    CN.columns = hydrovar_names
    return pd.concat([CN,hydrographs],ignore_index=True)


def NSE(Hmod,Hobs,obs_mean=media):
    #variables = Hmod.count +1
    """a = disc_time
    times = []
    for i,row in Hmod.iterrows():
        date = Hobs.iloc[0]['date']+datetime.timedelta(minutes = a)
        times.append(date)
        a = a +disc_time
    Hmod['time']=times"""
    Mod = Hmod.iloc[1:]
    Hobs=Hobs.reset_index(drop=True)
    #commom_data = Hobs[(Hobs.date.isin(Hmod.time))]
    nash =[]
    for i in range(len(Hmod.columns)):
        up = 0
        down = 0
        column = Mod[[i]]
        for j in range(len(column.index)):
            a=column.iloc[j][0]
            b=Hobs.iloc[j]['wo_base']
            up = up + np.square(a - b)
            down = down + np.square(Hobs.iloc[j]['wo_base'] - obs_mean)
        nash.append(np.abs(1-up/down))
    return(nash)

   
def Peak_Dif(Hmod,Hobs,max_obs):
    peak_dif =[]
    for i in range(len(Hmod.columns)-1):
        column = Hmod[[i]]
        max_modelled = column.max().reset_index(drop=True)
        max_modelled =max_modelled.iloc[[0]][0]
        diff = np.abs(max_modelled - max_obs)
        peak_dif.append(diff)
    return(np.abs(peak_dif))
   
def menorF1(x):
    a = x['F1'] - 1
    b = a.abs()
    c = b.sort_values()
    d = c[:1]
    return(x[x['F1'].index==d.index.values[0]])
    #a = x.ix[(x['F1']-1).abs().argsort()[:1]].reset_index(drop=True)
    #return a.iloc[:1,:]

def menorF2(x):
    return x[x['F2']==x['F2'].min()].reset_index(drop=True)



def ParetoFront(pop):
    pop_atualizado = pd.DataFrame(columns=['F1','F2','nome'])
    contador = 1
    a = []
    while len(pop.index)>0:
        sol_menorF1 = menorF1(pop)
        sol_menorF1  = sol_menorF1.drop_duplicates(subset='nome')
        sol_menorF2 = menorF2(pop).drop_duplicates(subset='F2')
        menoresF1 = pop[pop['F1']<sol_menorF2.iloc[0]['F1']].reset_index(drop=True)
        menoresF2 = pop[pop['F2']<sol_menorF1.iloc[0]['F2']].reset_index(drop=True)
        menoresF = menoresF1[menoresF1['nome'].isin(menoresF2['nome'])]
        pop_atualizado = pd.concat([pop_atualizado,sol_menorF1,sol_menorF2]).reset_index(drop=True)
        
        while len(menoresF.index)>=1:
            sol_menorF1 = menorF1(menoresF)
            sol_menorF2 = menorF2(menoresF)
            menoresF1 = pop[pop['F1']<sol_menorF2.iloc[0]['F1']].reset_index(drop=True)
            menoresF2 = pop[pop['F2']<sol_menorF1.iloc[0]['F2']].reset_index(drop=True)
            pop_atualizado = pd.concat([pop_atualizado,sol_menorF1,sol_menorF2]).reset_index(drop=True)
            menoresF = menoresF1[~menoresF1['nome'].isin(menoresF2['nome'])]    
        
        numerodafrente = pd.DataFrame(np.full((len(pop_atualizado.index), 1), contador),columns=['frente'])
        solucaocomfrente = pd.concat([pop_atualizado, numerodafrente], axis=1).reset_index(drop=True)
        #solucaocomfrente = solucaocomfrente[list(solucaocomfrente.columns[~solucaocomfrente.columns.duplicated()])]
        a.append(solucaocomfrente.drop_duplicates())
        pop = pop[~pop['nome'].isin(solucaocomfrente['nome'])]
        pop_atualizado = pd.DataFrame(columns=['F1','F2','nome'])
        contador = contador + 1
    df = pd.concat(a).reset_index(drop=True)
    return df

def CrowdDistance(pop_w_fronts,pop_size):
    #Calcula o valor dos cubóides    
    df = pop_w_fronts.sort_values(by=['F1'])
    df = df.sort_values(by=['frente']).reset_index(drop=True)
    b = df.groupby('frente').diff(periods=2)
    cuboide = np.abs(b['F1']/(maxF1-minF1)) + np.abs(b['F2']/(maxF2-minF2))
    cuboide = cuboide.rename('distancia')
    cuboide = cuboide.drop(labels=[0]).reset_index(drop=True)
    populacao_com_cuboides = pd.concat([df,cuboide],axis=1).reset_index(drop=True)
    
    nova_populacao = pd.DataFrame(columns=['F1','F2','nome','frente'])
    cont = 1
    #Seleção de melhores soluções:
    while len(nova_populacao.index) < pop_size:
        quantidade = pop_size - len(nova_populacao.index)
        a = populacao_com_cuboides.loc[populacao_com_cuboides['frente'] == cont]
        b = a.loc[pd.isnull(a['distancia'])]
        c = quantidade - len(a.index)
        d = quantidade - len(b.index)
        if c >= 0:
            nova_populacao = pd.concat([nova_populacao,a]).reset_index(drop=True)
        elif d > 0:
            q = quantidade - len(b.index)
            z = a.loc[a['distancia'].notnull()]
            z = z.sort_values(by=['distancia'],ascending=False)
            z = z[:q]
            nova_populacao = pd.concat([nova_populacao,b,z]).reset_index(drop=True)
        elif d == 0:
            nova_populacao = pd.concat([nova_populacao,b]).reset_index(drop=True)
        elif d < 0:
            q = quantidade
            b = b.ix[:q]
            nova_populacao = pd.concat([nova_populacao,b]).reset_index(drop=True)
            cont = cont + 1
    return(nova_populacao)

def Crossover(sol1,sol2,probability):
    r=random.random()
    if r > probability:
        sol_cross1 = pd.concat([sol1.iloc[:,0],sol2.iloc[:,1:7],sol1.iloc[:,8:]],axis=1)
        sol_cross2 = pd.concat([sol2.iloc[:,0],sol1.iloc[:,1:7],sol2.iloc[:,8:]],axis=1)
    else:
        sol_cross1 = sol1
        sol_cross2 = sol2
    return((sol_cross1,sol_cross2))

def Mutation(sol,probability):
    r=random.random()
    sol_mutable = sol.iloc[:,0:16]
    sol_rest=sol.iloc[:,16:]
    if r < probability:
        #sol = sol.select_dtypes(exclude=['object', 'string']) * 3
        #sol = sol_mutable.astype(float)
        sol_mut = sol_mutable.multiply(random.random())
        sol_mut = pd.concat([sol_mut,sol_rest],axis=1)
    else:
        sol_mut = sol
    return(sol_mut)

def CreateChild(pop_mother,input_data,commom_data,
                p_crossover=Pr,p_mutation=Pm,
                media=media,max_obs=max_obs,ptotal=ptotal,prec=prec,
                p_duration=p_duration,disc_interval=disc_interval):
    child = pop_mother

    sols = []
    sols_t=[]
    n=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'F1','F2','nome']
    del n[-1]
    del n[-1]
    del n[-1]
    child_solution = pd.DataFrame(columns=n)
    #Crossover
    aux = 0
    while aux < len(pop_mother.index):
        nome1 = pop_mother.iloc[aux]['nome']
        nome2 = pop_mother.iloc[aux+1]['nome']
        sol1 = input_data.loc[input_data['nome'] == nome1]
        sol2 = input_data.loc[input_data['nome'] == nome2]
        sol_cross1,sol_cross2=Crossover(sol1,sol2,p_crossover)
        sols_t.append(sol1)
        #child_solution=pd.concat([child_solution,sol_cross1,sol_cross2])
        sols.append(sol_cross1)
        sols.append(sol_cross2)
        aux = aux+2

    for i in sols:
        sol_mut = Mutation(i,p_mutation)
        child_solution=pd.concat([child_solution,sol_mut]).reset_index(drop=True)

    #return(child_solution.dropna().reset_index(drop=True))
    
    p_efetivas = []
    
    for i, row in child_solution.iterrows():
        CN = row[0]
        Pef = prec_efetiva(prec,CN,p_duration,disc_interval)
        p_efetivas.append(Pef)
    
    child_transposed = child_solution.iloc[:,0:16].T
    hydrographs = convolution(child_transposed,p_efetivas)
    """a = 0
    times = []
    for i,row in hydrographs.iterrows():
        if i == 0:
            times.append(np.NaN)
            a = a + 0
        else:
            date = new_hid.iloc[0]['date']+datetime.timedelta(minutes = a)
            times.append(date)
            a = a +disc_interval"""
            
    #hydrographs['time']=times
    nash = NSE(hydrographs,commom_data)
    peakdiff = Peak_Dif(child_transposed,commom_data,max_obs)
    
    
    
    """child_solution=child_transposed.append(pd.Series(nash, index=n), ignore_index=True)
    child_solution=child_solution.append(pd.Series(peakdiff, index=n), ignore_index=True)
    child_solution=child_solution.append(pd.Series(n, index=n), ignore_index=True)
    names = []
    for i in range(len(child_solution.index)):
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        names.append(name)
    child_solution['nome']=names"""
    
    child_solution= pd.concat([child_solution,pd.Series(nash)],axis=1, ignore_index=True)
    child_solution=pd.concat([child_solution,pd.Series(peakdiff)],axis=1, ignore_index=True)
    #child_solution=pd.concat([child_solution,pd.Series(n, index=n)],axis=1, ignore_index=True)
    names = []
    for i in range(len(child_solution.index)):
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        names.append(name)
    child_solution = child_solution.rename(columns={19: 'F1', 20: 'F2'})
    child_solution['nome']=names
                  
    return(child_transposed,child_solution)

#criando população inicial
input_data=create_pop(pop_i,variables)

CNs = input_data.iloc[0]

Pef = []

for index, CN in CNs.iteritems():
    Pef.append(prec_efetiva(prec,CN,p_duration,disc_interval))
    
hydrographs = convolution(input_data,Pef)

a = 0

times = []
           
for i,row in hydrographs.iterrows():
    if i == 0:
        times.append(np.NaN)
        a = a + 0
    else:
        date = new_hid.iloc[0]['date']+datetime.timedelta(minutes = a)
        times.append(date)
        a = a +disc_interval

hydrographs['time']=times
           
last_row_hydrographs = hydrographs.tail(1)
last_row_new_hid = new_hid.tail(1)
diff_between_ends = last_row_hydrographs.iloc[0]['time'] - last_row_new_hid.iloc[0]['date']
time_step_new_hid = new_hid.iloc[1]['date'] - new_hid.iloc[0]['date']
time_step_new_hid = time_step_new_hid.total_seconds()/60
                                                   
a = last_row_new_hid.iloc[0]['date']
dates = []
while a <= last_row_hydrographs.iloc[0]['time']:
    a = a +datetime.timedelta(minutes = time_step_new_hid)
    dates.append(a)
    
columns_names = list(new_hid.columns.values)
new_column = {}
x = np.zeros(len(dates), dtype=float)
for i in columns_names:
    if i == 'date':
        new_column[i]=dates
    else:
        new_column[i]=x
new_df = pd.DataFrame(new_column,columns=columns_names)
new_hid = pd.concat([new_hid,new_df],axis=0,ignore_index=True)

commom_data = new_hid[(new_hid.date.isin(hydrographs.time))]
commom_data = commom_data.reset_index(drop=True)

teste = hydrographs.iloc[1:,:-1].reset_index(drop=True)
#hydrographs = hydrographs.iloc[3:]
nash = NSE(hydrographs.iloc[1:,:-1].reset_index(drop=True),commom_data,media)

peak_dif = Peak_Dif(hydrographs,commom_data,max_obs)

names= list(input_data.columns.values)
input_data=input_data.append(pd.Series(nash, index=names), ignore_index=True)
input_data=input_data.append(pd.Series(peak_dif, index=names), ignore_index=True)
input_data=input_data.append(pd.Series(names, index=names), ignore_index=True)

input_data_transposed = input_data.T
input_data_transposed.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'F1','F2','nome']
pop = input_data_transposed.reset_index(drop=True)
pop = pop.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], axis=1)

#pop_mother,input_data,p_crossover,p_mutation,commom_data,media,max_obs,ptotal,prec,p_duration,
#                disc_interval)
    
#a = pd.DataFrame({'F1':np.array([4, 3, 0.1,2,6,0.1]).tolist()},columns=['F1'])
#b,c = menorF1(a)

solucoes = []
for i in range(GER):
    pop_w_fronts = ParetoFront(pop)        
    nova_populacao = CrowdDistance(pop_w_fronts,pop_i)
    nash,pop_filha = CreateChild(nova_populacao,input_data_transposed,commom_data)
    pop_filha[[0]]=pop_filha[[0]].fillna(1)
    pop_filha=pop_filha.fillna(0)
    input_data_transposed = pd.concat([input_data_transposed,pop_filha],ignore_index=True)
    pop_filha_wo_variables = pop_filha[['F1','F2','nome']]
    #pop_filha_wo_variables.columns = ['F1','F2','nome']
    nova_populacao_wo = nova_populacao[['F1','F2','nome']]
    pop = pd.concat([pop_filha_wo_variables,nova_populacao_wo],ignore_index=True)
#solucoes = solucoes.append(nova_populacao)
#iteracoes = pd.concat([solucoes],axis=1).reset_index(drop=True)
plt.figure()
plt.plot(pop_w_fronts['F1'], pop_w_fronts['F2'], 'o')
plt.ylabel('F1')
plt.ylabel('F2')
plt.show()