from requests.compat import *
from requests import request
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np

def recode_scraper(category1, category2, year):
    url = 'http://sporki.statiz.co.kr/stats/'
        
    if category1 == 'fielding':
        params = {
            'm':'main',
            'm2':category1,
            'm3':category2,
            'year':str(year),
            'pr':'1000',
            'ls':'1'
        }
    else:
        params = {
            'm':'main',
            'm2':category1,
            'm3':category2,
            'year':str(year),
            'pr':'1000',
            'reg':'C30',
            'ls':'1'
        }

    resp = request('GET',url=url, params=params)
    dom = BeautifulSoup(resp.text,'html.parser')
    
    data = [node.text for node in dom.find_all('td')]
    position_data = dom.select('div > span')
    
    players = [item.text for item in dom.select('td > div > a')]
    players_href = [tag.attrs['href'] for tag in dom.select('td> div > a')]
    players_num = [re.search(r'\d+', item).group() for item in players_href]
    attrs_2 = [item['title'] for item in dom.select('li[title]')]
    if (category1 == 'pitching' and category2 == 'pitch'):
        attrs_2[19] = 'Looking Strike Out, 루킹 삼진'
        attrs_2[20] = 'Swing Strike Out, 헛스윙 삼진'
    attrs = attrs_2[:int(len(attrs_2)/2)]
    columns = ['Number', '이름','Position'] + attrs
    df = pd.DataFrame(columns = columns)

    for i in range(len(players)):
        row = {}
        row['Number'] = players_num[i]
        row['이름'] = players[i]
        if (category1 == 'fielding'):
            row['Position'] = position_data[31+3*i].text
        else:
            row['Position'] = position_data[49+3*i].text
        for j in range(len(attrs)):
            index = i*(len(attrs)+4) + 4 + j
            row[attrs[j]] = data[index]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df

def get_position_code(position):
    """각 포지션에 대한 코드를 반환합니다."""
    if position == '1루수':
        return '1B'
    elif position == '2루수':
        return '2B'
    elif position == '3루수':
        return '3B'
    elif position == '유격수':
        return 'SS'
    elif position == '좌익수':
        return 'LF'
    elif position == '중견수':
        return 'CF'
    elif position == '우익수':
        return 'RF'
    elif position == '포수':
        return 'C'
    else:
        return None

def get_raa_sum(records_df, player_number, pos_code):
    try:
        specific_record = records_df[(records_df['Number'] == player_number) & (records_df['Position'] == pos_code)]
        raa = float(specific_record['Fielding Runs Above Average/144 Games, 평균 대비 수비 득점 기여 (144경기 기준)'].iloc[0])
    except (ValueError, IndexError):
        raa_list = list(map(float, records_df[records_df['Number'] == player_number]['Fielding Runs Above Average/144 Games, 평균 대비 수비 득점 기여 (144경기 기준)']))
        raa = sum(raa_list) / len(raa_list) if raa_list else 0
    return raa

def calculate_raa_sums(positions_num, outfield_recode, infield_recode, catcher_recode):
    outfield_raa_sum = infield_raa_sum = catcher_raa_sum = 0.0
    position_keys = list(positions_num.keys())

    for i, player_number in enumerate(positions_num.values()):
        position = position_keys[i]
#         print(position)
        pos_code = get_position_code(position)
#         print(pos_code)
#         print(player_number)
        
        if position in ['우익수', '좌익수', '중견수']:
            outfield_raa_sum += get_raa_sum(outfield_recode, player_number, pos_code)
        elif position in ['1루수', '2루수', '유격수', '3루수']:
            infield_raa_sum += get_raa_sum(infield_recode, player_number, pos_code)
        elif position in ['포수']:
            players_recode = catcher_recode[catcher_recode['Number'] == player_number]
            try:
                raa_include_framing = float(players_recode[players_recode['Position'] == 'C']['Fielding Runs Above Average/144 Games, 평균 대비 수비 득점 기여 (144경기 기준)'])
                raa_framing_144 = (float(players_recode[players_recode['Position'] == 'C']['Fielding Runs (Framing), 프레이밍 관련 득점 기여'])*144) / float(players_recode[players_recode['Position'] == 'C']['Games Played, 출장'])
                raa = raa_include_framing - raa_framing_144
                catcher_raa_sum = catcher_raa_sum + raa
            except:
                raa_list = list(map(float,list(players_recode['Fielding Runs Above Average/144 Games, 평균 대비 수비 득점 기여 (144경기 기준)'])))
                raa = sum(raa_list)/len(raa_list) if raa_list else 0
                catcher_raa_sum = catcher_raa_sum + raa

    return outfield_raa_sum, infield_raa_sum, catcher_raa_sum

def gamelog_scraper(year):
    
    games_schedule = list()
    
    # 1 ~ 12월까지 경기 ID 수집
    for i in range(1,13):
        url = 'http://sporki.statiz.co.kr/schedule/'
    
        params = {
            'year':str(year),
            'month':str(i)
        }
        
        resp = request('GET',url=url, params=params)
        dom = BeautifulSoup(resp.text,'html.parser')
        
        if (['p_schedule'] in list(dom.select('div > ul > li >a')[-1].attrs.values())):
            print('skip')
            continue
        
        print(i)
        games_tag = [tag for tag in dom.select('div > ul > li >a') if 'class' not in list(tag.attrs.keys())]
        games_href = [tag.attrs['href'] for tag in games_tag]
        games_num = [re.search(r'\d+', item).group() for item in games_href]
        
        games_schedule = games_schedule + games_num
    
    # Gamelog feature 지정
    columns = ['날짜', '경기장', 'Pitcher Number','Hitter Number','회', '내야 수비 RAA','외야 수비 RAA','포수 수비 RAA', '투수 이름', 
                '타자 이름', '투구수', '결과', '이전상황','이후상황','LEV','REs','REa','WPe','WPa']
    df = pd.DataFrame(columns = columns)
    
    # 수비 데이터 호출
    outfield_recode = recode_scraper('fielding','outField',year)
    infield_recode = recode_scraper('fielding','inField',year)
    catcher_recode = recode_scraper('fielding','catcher',year)
    
    # 게임 별 Gamelog 수집
    for game_id in games_schedule:
        
        url = 'http://sporki.statiz.co.kr/schedule/'
        
        params = {
            'm':'gamelogs',
            's_no':game_id
        }

        resp = request('GET',url=url, params=params)
        dom = BeautifulSoup(resp.text,'html.parser')
        
        stadium = dom.find_all(attrs={'class':'txt'})[1].text[7:9]
        date = dom.find_all(attrs={'class':'txt'})[1].text[11:16]
        print(stadium, date)
        game_canceled = dom.find_all(attrs={'style':'font-size:1.6rem;'})
    
        if (len(game_canceled) > 0):
            continue
        
        data = [node.text for node in dom.find_all('td')]
        lineup_data = data[:80]
        data = data[80:-4]
        
        players = [item.text for item in dom.select('td > a')]
        players_href = [tag.attrs['href'] for tag in dom.select('td > a')]
        players_num = []

        for item in players_href:
            match = re.search(r'\d+', item)
            if match:
                players_num.append(match.group())
            else:
                players_num.append('')
                
        players_num_lineup = players_num[:20]
        lineup_columns = ['Player Number', 'Name', 'Position', 'Bats/Throws', 'Home/Away']
        lineup_df = pd.DataFrame(columns=lineup_columns)

        home_away_toggle = 'Away'  # 초기 팀을 'Away'로 설정
        last_number = None

        for i in range(0, len(lineup_data), 4):
            number = lineup_data[i]
            name = lineup_data[i + 1]
            position = lineup_data[i + 2]
            bats_throws = lineup_data[i + 3]

            # 번호가 '1'로 재시작하면 Home/Away 토글
            if number == '1' and last_number is not None:
                home_away_toggle = 'Away' if home_away_toggle == 'Home' else 'Home'

            player_data = pd.DataFrame({
                'Name': [name],
                'Position': [position],
                'Bats/Throws': [bats_throws],
                'Home/Away': [home_away_toggle]
            })
            lineup_df = pd.concat([lineup_df, player_data], ignore_index=True)
            last_number = number

        for i in range(len(players_num_lineup)):
            number = players_num_lineup[i]
            lineup_df['Player Number'][i] = number
            
        # Starting lineup에서 Position 연결
        positions = ['좌익수', '중견수', '우익수', '1루수', '2루수', '유격수', '3루수','포수']
        positions_num_home = {}
        positions_num_away = {}

        df_positions = lineup_df[(lineup_df['Position'].isin(positions))]
        i = 0
        j = 0

        for home_player_position in list(df_positions[df_positions['Home/Away'] == 'Home']['Position']):
            positions_num_home[home_player_position] = list(df_positions[df_positions['Home/Away'] == 'Home']['Player Number'])[i]
            i = i+1
        for away_player_position in list(df_positions[df_positions['Home/Away'] == 'Away']['Position']):
            positions_num_away[away_player_position] = list(df_positions[df_positions['Home/Away'] == 'Away']['Player Number'])[j]
            j = j+1
        
        # Home/Away 수비 RAA 산출
        home_outfield_raa_sum, home_infield_raa_sum, home_catcher_raa_sum = calculate_raa_sums(positions_num_home, outfield_recode, infield_recode, catcher_recode)
        away_outfield_raa_sum, away_infield_raa_sum, away_catcher_raa_sum = calculate_raa_sums(positions_num_away, outfield_recode, infield_recode, catcher_recode)
        
        # Gamelog 생성
        for i in range(int(len(data)/12)):
            row = {}
            
            row['날짜'] = date
            row['경기장'] = stadium
            row['Pitcher Number'] = players_num[2*i+20]
            row['Hitter Number'] = players_num[2*i + 21]
            if data[12*i] != '':
                row['회'] = data[12*i]
                inning = data[12*i]
            else:
                row['회'] = inning
                
            # 수비 RAA 지정
            if row['회'][-1] == '초':
                row['내야 수비 RAA'] = home_infield_raa_sum
                row['외야 수비 RAA'] = home_outfield_raa_sum
                row['포수 수비 RAA'] = home_catcher_raa_sum
            else:
                row['내야 수비 RAA'] = away_infield_raa_sum
                row['외야 수비 RAA'] = away_outfield_raa_sum
                row['포수 수비 RAA'] = away_catcher_raa_sum 

            for j in range(11):
                index = i*12 + 1 + j
                row[columns[j+8]] = data[index]

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df