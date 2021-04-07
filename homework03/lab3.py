#!/usr/bin/env python
# coding: utf-8

# Отметим, что данные нашего датасета немного не совпадают с данными на сайте ЦИК.\
# Расхождения замечены в нескольких регионах. Как мы это учтём:\
#     1. При анализе выборов по регионам заменим часть датасета на данные с сайта ЦИК там, где замечены несовпадения.\
#     2. При дальнейшем анализе данных регионов по участкам будем иметь в виду, что наша информация немного неполная/неактуальная.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv('../voting_data.csv')
df


# Добавим в таблицу информацию об общем количестве голосов на участке

# In[3]:


voted_list = []
turnout_list = []

for _, row in df.iterrows():
    voted = row['N_EARLY'] + row['N_IN'] + row['N_OUT']
    voted_list.append(voted) # or N_GIVEN - N_LEFT
    turnout_list.append(voted/row['N_ALL'])
    
df['N_VOTED'] = voted_list
df['TURNOUT'] = turnout_list
df


# Посмотрим, какие столбцы у нас есть

# In[4]:


list(df.columns)


# Заметим, что в таблице много столбцов с избыточными/ненужными данными, а именно:\
# N_GIVEN, N_LEFT - могут использоваться только для подсчёта N_VOTED, что мы уже сделали\
# N_IN - вряд ли будем использовать, но в случае необходимости посчитаем как N_VOTED - N_OUT - N_EARLY\
# N_PORTABLE, N_STATIC - почти полностью совпадают с N_OUT и N_IN, нет необходимости их хранения\
# N_VALID - можно посчитать как N_VOTED - N_INVALID\
# N_UNUSED, N_LOST - почти всегда равны 0 (см. ниже) и не влияют на общую картину

# In[5]:


df['N_UNUSED'].sum()


# In[6]:


df['N_LOST'].sum()


# Покажем корреляцию столбцов

# In[7]:


corr = df.corr()
sns.heatmap(corr, cmap="BuGn", vmin=0)


# Рассмотрим ближе интересующие нас стобцы

# In[8]:


corr = df[['N_IN', 'N_VOTED', 'N_VALID', 'N_STATIC']].corr()
sns.heatmap(corr, cmap="BuGn", vmin=0)


# In[9]:


corr = df[['N_OUT', 'N_PORTABLE']].corr()
sns.heatmap(corr, cmap="BuGn", vmin=0)


# Удалим ненужные столбцы после создания таблицы с результатами выборов по регионам

# У нас есть два региона, названия которых в датасете начинаются с цифры. Для удобства эти цифры удалим

# In[10]:


def remove_digits(text):
    if text[0].isdigit():
        temp = list(map(lambda x: '' if x.isdigit() else x, text))
        text = ''.join(temp).strip()
    return text


# In[11]:


df['REGION'] = df['REGION'].apply(remove_digits)
df['SUBREGION'] = df['SUBREGION'].apply(remove_digits)
df['REGION'].unique()[:2]


# Объявим некоторые константы, которые пригодятся при анализе данных

# In[12]:


start = list(df.columns).index('BABURIN')
end = list(df.columns).index('YAVLINSKY') + 1
CANDS = [i for i in df.columns[start:end]]
COLORS = ['lightsalmon', 'red', 'lightblue', 'CornflowerBlue', 'olive', 'yellow', 'purple', 'lightgreen']
DF_LIST = [i for i in range(df.shape[0])]
INCOMPLETE_REGIONS = ['Ханты-Мансийский автономный округ - Югра', 'Республика Алтай', 'Республика Дагестан',
               'Чувашская Республика - Чувашия', 'Брянская область', 'Волгоградская область',
               'Калининградская область', 'Кемеровская область', 'Магаданская область',
               'Нижегородская область', 'город Москва', 'Территория за пределами РФ']


# Представим результаты выборов на круговой диаграмме

# In[13]:


labels = CANDS.copy() + ['INVALID']
start = list(df.columns).index('BABURIN')
end = list(df.columns).index('YAVLINSKY') + 1
sizes = [df[i].sum() for i in df.columns[start:end]]
sizes.append(df['N_INVALID'].sum())

_, ax = plt.subplots()
ax.pie(sizes, labels=labels,
        autopct=(lambda x: f'{x:.2f} %' if x > 5.0 else ''),
        shadow=True, startangle=90, labeldistance=None,
        colors=COLORS + ['grey'])
ax.axis('equal')
ax.legend(labels, title="Candidates", loc="center", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Election results')
plt.show()


# На диаграмме мы видим распределение голосов на выборах. Отметим, что показанные на диаграмме проценты соответствуют официальным данным (наши недостающие данных составляют слишком малую часть, поэтому на общие итоги не влияют)

# In[14]:


for k in range(len(CANDS)):
    plt.plot(DF_LIST, [df[CANDS[k]][i] for i in range(df.shape[0])], '.', color=COLORS[k])
    
plt.title('Total number of votes by polling station')
plt.show()


# На диаграмме выше представлено количество голосов за каждого кандидата по УИК. Как мы видим, за Путина в среднем голосовало по несколько тысяч человек на участке, в то время как за других кандидатов - явно меньше тысячи. Чтобы подробнее увидеть информацию о других кандидатах, уберём голоса за победителя

# In[15]:


for k in range(len(CANDS)):
    if k == CANDS.index('PUTIN'): continue
    plt.plot(DF_LIST, [df[CANDS[k]][i] for i in range(df.shape[0])], '.', color=COLORS[k])
    
plt.title('Total number of votes by polling station without the winner')
plt.show()


# По данной диаграмме видно, что на втором месте с достаточным отрывов находится Грудинин. В районе 17000-х участков можно заметить резкий рост голосов за Сурайкина. Возможно, в дальнейшем остановимся на этом подробнее
# 
# Заметим, что по анализу отдельных УИК тяжело сделать какие-либо выводы, кроме самых очевидных. Будем проводить анализ по регионам

# Создадим таблицу с данным по регионам. Для этого просуммируем данные из всех участков данного региона и запишем их в одну строку. Для регионов, где в первой таблице содержатся неполные данные, возьмём данные с сайта ЦИК

# In[25]:


try:
    df_regions = pd.read_csv('regions_data.csv', index_col=0)
except FileNotFoundError:
    df_regions = pd.DataFrame(columns=df.columns)
    df_regions = df_regions.drop(columns=['PS_ID', 'SUBREGION', 'N_VOTED', 'TURNOUT'])

    for region in df['REGION'].unique():
        info = {}
        new_region = remove_digits(region)

        info['REGION'] = new_region
        for col in df.columns[3:-2]:
            info[col] = df[df['REGION'] == region][col].sum()
        df_regions = df_regions.append(info, ignore_index=True)

    for region in INCOMPLETE_REGIONS:
        new_info = []
        data = pd.read_csv('../' + region + '.csv')
        for row in data['Unnamed: 2']:
            try:
                new_info.append(int(row))
            except ValueError:
                continue

        index = int(df_regions[df_regions['REGION'] == region].index.values)
        for i, col in enumerate(df_regions.columns[1:]):
            df_regions.at[index, col] = new_info[i+2]
            
df_regions


# Добавим в таблицу данные об общем количестве проголосовавших и явке

# In[26]:


turnout_list = []
voted_list = []

for _, row in df_regions.iterrows():
    voted = row['N_EARLY'] + row['N_IN'] + row['N_OUT']
    voted_list.append(voted)
    turnout_list.append(voted/row['N_ALL'])
    
df_regions['N_VOTED'] = voted_list
df_regions['TURNOUT'] = turnout_list
df_regions.to_csv('regions_data.csv')
df_regions


# Как и в исходном датасете, здесь нам не нужны некоторые столбцы. Удалим их в обеих таблицах

# In[27]:


df = df.drop(columns=['N_GIVEN', 'N_LEFT', 'N_IN', 'N_PORTABLE', 'N_STATIC', 'N_VALID', 'N_UNUSED', 'N_LOST'], errors='ignore')
df_regions = df_regions.drop(columns=['N_GIVEN', 'N_LEFT', 'N_IN', 'N_PORTABLE', 'N_STATIC', 'N_VALID', 'N_UNUSED', 'N_LOST'], errors='ignore')
DF_REGIONS_LIST = [i for i in range(df_regions.shape[0])]
df_regions


# In[28]:


plt.plot(DF_REGIONS_LIST, turnout_list, '.', color='orange')
plt.title('Turnout by region')
plt.show()


# На графике показано распределение явки по регионам. Видно, что во всех регионах явка больше половину, в некоторых - почти 100 % (к этим регионам ещё вернёмся)

# In[29]:


turnout_list = [df_regions['N_VOTED'][i]/df_regions['N_ALL'][i] for i in range(df_regions.shape[0])]
plt.plot(DF_REGIONS_LIST, sorted(turnout_list), '.', color='orange')
plt.title('Turnout by region (sorted)')
plt.show()


# Для удобства отсортировали график явки по регионам. Видим, что в большинстве регионов явка составила около 65 %

# In[30]:


plt.plot(DF_REGIONS_LIST, [df_regions['N_EARLY'][i]/df_regions['N_VOTED'][i] for i in range(df_regions.shape[0])], '.')
plt.title('Early votes by region')
plt.show()


# На графике выше - процент голосов до дня голосования. В среднем он почти нулевой, однако есть регионы,
# где он составляет 5, 10 и даже 25 процентов

# In[31]:


plt.plot(DF_REGIONS_LIST, [df_regions['N_OUT'][i]/df_regions['N_VOTED'][i] for i in range(df_regions.shape[0])], '.')
plt.title('Votes outside the station by region')
plt.show()


# На данном графике - процент голосов за пределами УИК. Видим, что точки на графике распредены почти равномерно

# Добавим функцию для изображения линии, равной среднему проценту победителя в выборах. Будем её использовать во многих графиках

# In[32]:


def add_winner_average_line(obj):
    putin_list = [df_regions['PUTIN'][i]/df_regions['N_VOTED'][i] for i in range(df_regions.shape[0])]
    obj.axhline(y=sum(putin_list)/len(putin_list), ls='--', color=COLORS[CANDS.index('PUTIN')])


# In[33]:


putin_list = [df_regions['PUTIN'][i]/df_regions['N_VOTED'][i] for i in range(df_regions.shape[0])]

turnout_sorted, putin_sorted = zip(*sorted(zip(turnout_list, putin_list)))
plt.plot(DF_REGIONS_LIST, turnout_sorted, '.', color='orange')
plt.plot(DF_REGIONS_LIST, putin_sorted, '.', color=COLORS[CANDS.index('PUTIN')])    
    
add_winner_average_line(plt)
plt.title('Dependence of votes for Putin on turnout')
plt.show()


# На графике представлена зависимость голосов за Путина от явки. Как мы видим, наибольший процент голосов за Путина - в регионах с максимальной явкой. Более того, среди регионов с явкой меньше 75 % нет почти ни одного, где процент голосов за Путина превышал бы средний. Подозрительно. Далее изучим подробнее регионы с самой большой явкой (и низкой тоже). Для этого отсортируем нашу таблицу по явке

# In[34]:


df_regions_sorted = df_regions.sort_values(by='TURNOUT', ascending=False, ignore_index=True)
df_regions_sorted


# Создадим функцию отображения явки и голосов за Путина в виде столбчатой диаграммы, чтобы подробнее проанализировать интересующие нас регионы

# In[35]:


def draw_turnout_bar(num, reverse=False):
    if num > 10:
        print('The maxmium number of regions is 10')
        return
    
    start = 0 if not reverse else df_regions_sorted.shape[0] - 1
    stop = num if not reverse else start - num
    step = 1 if not reverse else -1
    turnout_num_list = [df_regions_sorted['TURNOUT'][i] for i in range(start, stop, step)]
    putin_num_list = [df_regions_sorted['PUTIN'][i]/df_regions_sorted['N_VOTED'][i] for i in range(start, stop, step)]
    region_num_list = []
    for i in range(start, stop, step):
        new_region = ''
        region = df_regions_sorted['REGION'][i]
        for j in range(0, len(region), 15 - num):
            new_region += region[j:j + 15 - num] + '\n'   
        region_num_list.append(new_region)

    ind = np.arange(len(turnout_num_list))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(ind - width/2, turnout_num_list, width,
                    label='Turnout', color='orange')
    ax.bar(ind + width/2, putin_num_list, width,
                    label='Putin', color=COLORS[CANDS.index('PUTIN')])

    fig.tight_layout(pad=0.1)
    ax.set_title('Turnout and votes for Putin by region')
    ax.set_xticks(ind)
    ax.set_xticklabels(region_num_list, fontdict={'fontsize': 10})
    ax.legend(['Turnout', 'Putin'], loc="center", bbox_to_anchor=(1, 0, 0.5, 1))
    add_winner_average_line(ax)


# In[36]:


draw_turnout_bar(10)


# На диаграмме видно, что среди первых десяти регионах по явке нет ни одного, где процент за Путина был бы ниже среднего

# In[37]:


draw_turnout_bar(10, reverse=True)


# Здесь же, наоборот, нет ни одного региона, где количество голосов за Путина превышало бы среднее. Что ж, рассмотрим данные регионы ещё подробнее

# Создадим функцию для визуализации по региону, которая вызывает ещё три - каждая рисует один график

# In[38]:


def visualize_region_results(region):
    df_region = df.loc[df['REGION'] == region]
    if not len(df_region):
        print('You entered incorrect region')
        return
    if region in INCOMPLETE_REGIONS:
        print('Note that the data about this region is incomplete')
    df_region = df_region.reset_index()
    visualize_turnout(df_region)
    visualize_winner(df_region)
    visualize_dependence(df_region)


# In[39]:


def visualize_turnout(df_region):
    plt.plot([i for i in range(df_region.shape[0])], sorted([df_region['N_VOTED'][i]/df_region['N_ALL'][i] for i in range(df_region.shape[0])]), '.', color='orange')
    plt.title(f"Turnout in {df_region['REGION'][0]} by polling station")
    plt.show()


# In[40]:


def visualize_winner(df_region):
    plt.plot([i for i in range(df_region.shape[0])], sorted([df_region['PUTIN'][i]/df_region['N_VOTED'][i] for i in range(df_region.shape[0])]), '.', color=COLORS[CANDS.index('PUTIN')])
    plt.title(f"Votes for Putin in {df_region['REGION'][0]} by polling station")
    plt.show()


# In[41]:


def visualize_dependence(df_region):
    turnout_list = [df_region['N_VOTED'][i]/df_region['N_ALL'][i] for i in range(df_region.shape[0])]
    putin_list = [df_region['PUTIN'][i]/df_region['N_VOTED'][i] for i in range(df_region.shape[0])]
    turnout_sorted, putin_sorted = zip(*sorted(zip(turnout_list, putin_list)))
    plt.plot([i for i in range(df_region.shape[0])], turnout_sorted, '.', color='orange')   
    plt.plot([i for i in range(df_region.shape[0])], putin_sorted, '.', color=COLORS[CANDS.index('PUTIN')])   
    add_winner_average_line(plt)

    plt.title(f"Dependence of votes for Putin on turnout\nin {df_region['REGION'][0]} by polling station")
    plt.show()


# In[42]:


visualize_region_results(df_regions_sorted['REGION'][0])


# Как мы видим, при голосовании за пределами РФ явка составляет почти 100 % на большинстве участков. Это, скорее всего, связано с тем, что на участках за пределами РФ регистрируются заранее. Есть человек подал заявление на включение в список избирателей, он наверняка сходит и на выборы

# In[43]:


visualize_region_results(df_regions_sorted['REGION'][1])


# Видим, что даже в пределах региона нет ни одного участка, где процент у Путина был меньше среднего (как, например, на территории за пределами РФ)

# In[44]:


visualize_region_results(df_regions_sorted['REGION'][2])


# Заметим, что скопление точек у линии среднего значения именно там, где низкая явка

# In[45]:


visualize_region_results(df_regions_sorted['REGION'][3])


# Невероятно высокий процент голосов за Путина вне зависимости от явки (которая, к слову, тоже сосредоточена в районе 90 %)

# In[46]:


visualize_region_results(df_regions_sorted['REGION'][4])


# Знакомая ситуация: низкий процент у Путина там же, где низкая явка. И это при том, что на прошлых выборах в данном регионе Путин получил почти 100 % при почти стопроцентной явке. Кажется: в этот раз на нескольких участках были наблюдатели (и испортили всю картину!)

# Теперь посмотрим на регионы с низкой явкой

# In[47]:


for i in range(1, 6):
    visualize_region_results(df_regions_sorted['REGION'][df_regions.shape[0]-i])


# Заметим, что во всех пяти регионах как явка, так и голоса за Путина распределены более-менее равномерно. Голоса расположены по обе стороны от средней линии, независимо от явки

# Рассмотрим регион из середины списка. Получаем аналогичные результаты

# In[48]:


visualize_region_results(df_regions['REGION'][40])


# Посмотрим, что будет, если посчитать результаты выборов в той половине регионов, где низкая явка

# In[49]:


labels = CANDS.copy() + ['INVALID']
start = list(df_regions_sorted.columns).index('BABURIN')
end = list(df_regions_sorted.columns).index('YAVLINSKY') + 1
sizes = [df_regions_sorted[45:][i].sum() for i in df_regions_sorted.columns[start:end]]
sizes.append(df_regions_sorted['N_INVALID'][45:].sum())

_, ax = plt.subplots()
ax.pie(sizes, labels=labels,
        autopct=(lambda x: f'{x:.2f} %' if x > 5.0 else ''),
        shadow=True, startangle=90, labeldistance=None,
        colors=COLORS + ['grey'])
ax.axis('equal')
ax.legend(labels, title="Candidates", loc="center", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Election results without regions with the highest turnout')
plt.show()


# Видим, что Путин потерял больше трёх процентов. Хотя, казалось бы, люди должны голосовать примерно равномерно (во всяком случае, распределение голосов может быть обусловлено осособенностями региона, а не явкой)

# In[50]:


df.to_csv('voting_data_result.csv')
df_regions.to_csv('regions_data_result.csv')

