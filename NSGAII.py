import numpy as np
import pandas as pd

#Criando as funções objetivos

functions = {
    'F1': lambda x1: x1,
    'F2': lambda x1, x2: (1+ x2)/x1,
}
def menorF1(x):
    return x[x['F1']==x['F1'].min()].reset_index(drop=True)

def menorF2(x):
    return x[x['F2']==x['F2'].min()].reset_index(drop=True)

data = {
  "nome": [1,2,3,4,5,6,"a","b","c","d","e","f"],
  "F1":[0.31,0.43,0.22,0.59,0.66,0.83,0.21,0.79,0.51,0.27,0.58,0.24],
  "F2": [6.1,6.79,7.09,7.85,3.65,4.23,5.9,3.97,6.51,6.93,4.52,8.54]
}

maxF1 = 1 #Valor máximo da função objetivo 1
maxF2 = 60 #Valor máximo da função objetivo 2
minF1 = 0.1 #Valor mínimo da função objetivo 1
minF2 = 0 #Valor mínimo da função objetivo 2
npop = 6 #Número de indivíduos da população
solucoes = []

for i in range(3):
    a = []
    pop_atualizado = pd.DataFrame(columns=['nome','F1','F2'])
    pop = pd.read_csv('data.csv') #Criando população
    #Constrói a frente pareto
    contador = 1
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
        pop_atualizado = pd.DataFrame(columns=['nome','F1','F2'])
        contador = contador + 1
    
    #Calcula o valor dos cubóides    
    df = pd.concat(a).reset_index(drop=True)
    df = df.sort_values(by=['F1'])
    df = df.sort_values(by=['frente']).reset_index(drop=True)
    b = df.groupby('frente').diff(periods=2)
    cuboide = np.abs(b['F1']/(maxF1-minF1)) + np.abs(b['F2']/(maxF2-minF2))
    cuboide = cuboide.rename('distancia')
    cuboide = cuboide.drop(labels=[0]).reset_index(drop=True)
    populacao_com_cuboides = pd.concat([df,cuboide],axis=1).reset_index(drop=True)

    nova_populacao = pd.DataFrame(columns=['nome','F1','F2','frente'])
    cont = 1

    #Seleção de melhores soluções:
    while len(nova_populacao.index) < npop:
        quantidade = npop - len(nova_populacao.index)
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
    
    #Sorteio
    for i in range(npop):
        sorteio = nova_populacao.sample(n=2)
        distancias_infinitas = sorteio.loc[pd.isnull(sorteio['distancia'])]
        #Caso só tenha uma solução com perímetro do cubóide infinito, ela é selecionada:
        if len(distancias_infinitas.index) == 1:
            nova_populacao = pd.concat([nova_populacao,distancias_infinitas]).reset_index(drop=True)
        #Caso tenham duas, é selecionada uma de maneira aleatória:
        elif len(distancias_infinitas.index) == 2:
            selecao_aleatorio = distancias_infinitas.sample(n=1)
            nova_populacao = pd.concat([nova_populacao,selecao_aleatorio]).reset_index(drop=True)
        #Caso não tenha nenhuma, a melhor solução é selecionada conforme o valor do perímetro do cubóide:
        else:
            ordenado = sorteio.sort_values(by=['distancia'],ascending=False)
            maior_distancia = ordenado[:1]
            nova_populacao = pd.concat([nova_populacao,maior_distancia]).reset_index(drop=True)
    nova_populacao[['nome','F1','F2']].to_csv('data.csv')
    solucoes.append(nova_populacao)
    
iteracoes = pd.concat(solucoes,axis=1).reset_index(drop=True)
iteracoes.to_csv('solucao.csv')