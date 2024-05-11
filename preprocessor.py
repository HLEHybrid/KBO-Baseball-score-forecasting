from requests.compat import *
from requests import request
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from scraper import recode_scraper, gamelog_scraper

def bat_recode(year):
    batting_bat = recode_scraper('batting','bat',year)
    batting_bat['Hitter Number'] = batting_bat['Number']
    batting_bat['타자 이름'] = batting_bat['이름']
    batting_bat = batting_bat[['Hitter Number','타자 이름','Batting Average on Balls In Play, 인플레이 타구의 안타 비율','땅볼%','내야 뜬볼%',
                                     '외야 뜬볼%','뜬볼%','라인드라이브%','홈런 / 뜬볼%','내야안타%']]
    
    batting_direction = recode_scraper('batting','direction',year)
    batting_direction['Hitter Number'] = batting_direction['Number']
    batting_direction['타자 이름'] = batting_direction['이름']
    batting_direction = batting_direction[['Hitter Number','타자 이름','전체 좌측 타구 비율','전체 좌중앙 타구 비율','전체 중앙 타구 비율',
                                               '전체 우중앙 타구 비율','전체 우측 타구 비율']]
    
    batting_pitch = recode_scraper('batting','pitch',year)
    batting_pitch['Hitter Number'] = batting_pitch['Number']
    batting_pitch['타자 이름'] = batting_pitch['이름']
    batting_pitch = batting_pitch[['Hitter Number','타자 이름','Strike%, 전체 투구 대비 스트라이크', 'Called Strike%, 전체 투구 대비 루킹 스트라이크%',
       'Whiff%, 전체 투구 대비 헛스윙 스트라이크%', 'CSW%, 전체 투구 대비 루킹+헛스윙 스트라이크%',
       'Swing%, 스윙 비율', '스윙 대비 콘택트 비율', '스윙 대비 헛스윙 비율', '초구 스트라이크 비율',
       '초구 스윙 비율', '투스트라이크 카운트 투구 대비 삼진 결정 비율', 'Strike Zone%, 존 안에 들어온 투구 비율',
       'Strike Zone Swing%, 존 안에 들어온 투구 대비 스윙 비율',
       'Strike Zone Contact%, 존 안에 들어온 투구 대비 콘택트 비율',
       'Out Zone%, 존 밖에 들어온 투구 비율', 'Out Zone Swing%, 존 밖에 들어온 투구 대비 스윙 비율',
       'Out Zone Contact%, 존 밖에 들어온 투구 대비 콘택트 비율',
       'Meatball Zone%, 존 한가운데 들어온 투구 비율',
       'Meatball Swing%, 존 한가운데 들어온 투구 대비 스윙 비율',
       'Shadow Zone%, 쉐도우존에 들어온 투구 비율', 'Looking Strike Out%, 루킹 삼진 비율']]
    
    batting_balltype = recode_scraper('batting','ballType',year)
    batting_balltype = balltype_value_divider(batting_balltype, 'batting')

    batting_player_data = pd.merge(batting_bat,batting_direction)
    batting_player_data = pd.merge(batting_pitch,batting_player_data)
    batting_player_data = pd.merge(batting_balltype,batting_player_data)

    batting_split_data = []

    for id in list(batting_player_data['Hitter Number']):
        batting_split_num = split_finder(id,'batting')
        batting_split_data.append(batting_split_num)

    batting_player_data['투타'] = batting_split_data

    return batting_player_data


def pitch_recode(year):
    pitching_balltype = recode_scraper('pitching','ballType',year)
    pitching_balltype = balltype_value_divider(pitching_balltype, 'pitching')
    
    pitching_pitch = recode_scraper('pitching','pitch',2023)
    pitching_pitch.columns = ['(P) ' + column for column in pitching_pitch.columns]
    pitching_pitch['Pitcher Number'] = pitching_pitch['(P) Number']
    pitching_pitch['투수 이름'] = pitching_pitch['(P) 이름']
    pitching_pitch = pitching_pitch[['Pitcher Number','투수 이름', '(P) Strike%, 전체 투구 대비 스트라이크', '(P) Called Strike%, 전체 투구 대비 루킹 스트라이크%',
       '(P) Whiff%, 전체 투구 대비 헛스윙 스트라이크%', '(P) CSW%, 전체 투구 대비 루킹+헛스윙 스트라이크%',
       '(P) Swing%, 스윙 비율', '(P) 스윙 대비 콘택트 비율', '(P) 스윙 대비 헛스윙 비율',
       '(P) 초구 스트라이크 비율', '(P) 초구 스윙 비율', '(P) 투스트라이크 카운트 투구 대비 삼진 결정 비율',
       '(P) Strike Zone%, 존 안에 들어온 투구 비율',
       '(P) Strike Zone Swing%, 존 안에 들어온 투구 대비 스윙 비율',
       '(P) Strike Zone Contact%, 존 안에 들어온 투구 대비 콘택트 비율',
       '(P) Out Zone%, 존 밖에 들어온 투구 비율',
       '(P) Out Zone Swing%, 존 밖에 들어온 투구 대비 스윙 비율',
       '(P) Out Zone Contact%, 존 밖에 들어온 투구 대비 콘택트 비율',
       '(P) Meatball Zone%, 존 한가운데 들어온 투구 비율',
       '(P) Meatball Swing%, 존 한가운데 들어온 투구 대비 스윙 비율','(P) Looking Strike Out%, 루킹 삼진 비율']]
    
    pitching_direction = recode_scraper('pitching','direction',year)
    pitching_direction.columns = ['(P) ' + column for column in pitching_direction.columns]
    pitching_direction['Pitcher Number'] = pitching_direction['(P) Number']
    pitching_direction['투수 이름'] = pitching_direction['(P) 이름']
    pitching_direction = pitching_direction[['Pitcher Number','투수 이름', '(P) Left ball%, 전체 좌측 타구 비율',
       '(P) Left center ball%, 전체 좌중앙 타구 비율', '(P) Center ball%, 전체 중앙 타구 비율',
       '(P) Right center ball%, 전체 우중앙 타구 비율', '(P) Right ball%, 전체 우측 타구 비율',
       '(P) Pull-side ball%, 전체 당겨친 타구 비율']]
    
    pitching_player_data = pd.merge(pitching_balltype,pitching_pitch)
    pitching_player_data = pd.merge(pitching_direction,pitching_player_data)

    pitching_split_data = []

    for id in list(pitching_player_data['Pitcher Number']):
        pitching_split_num = split_finder(id,'pitching')
        pitching_split_data.append(pitching_split_num)

    pitching_player_data['(P) 투타'] = pitching_split_data
    
    return pitching_player_data



def balltype_value_divider(data, type):
    
    if type == 'batting':
        data['Hitter Number'] = data['Number']
        data['타자 이름'] = data['이름']

        balltype_list = ['구종가치 (투심)',
               '구종가치 (포심)', '구종가치 (커터)', '구종가치 (커브)', '구종가치 (슬라이더)', '구종가치 (체인지업)',
               '구종가치 (싱커)', '구종가치 (포크볼)']
        balltype_throw_list = ['구종별 투구 수 (투심)',
           '구종별 투구 수 (포심)', '구종별 투구 수 (커터)', '구종별 투구 수 (커브)', '구종별 투구 수 (슬라이더)',
           '구종별 투구 수 (체인지업)', '구종별 투구 수 (싱커)', '구종별 투구 수 (포크볼)']
        balltype_100_list = ['구종가치/100 (투심)',
           '구종가치/100 (포심)', '구종가치/100 (커터)', '구종가치/100 (커브)', '구종가치/100 (슬라이더)',
           '구종가치/100 (체인지업)', '구종가치/100 (싱커)', '구종가치/100 (포크볼)']
    elif type == 'pitching':
        data['Pitcher Number'] = data['Number']
        data['투수 이름'] = data['이름']
        
        balltype_list = ['2-Seamer Fastball Pitch Value, 구종가치 (투심)',
       '4-Seamer Fastball Pitch Value, 구종가치 (포심)',
       'Cutter Pitch Value, 구종가치 (커터)', 'Curve Pitch Value, 구종가치 (커브)',
       'Slider Pitch Value, 구종가치 (슬라이더)', 'Changeup Pitch Value, 구종가치 (체인지업)',
       'Sinker Pitch Value, 구종가치 (싱커)', 'Forkball Pitch Value, 구종가치 (포크볼)']
        balltype_throw_list = ['2-Seamer Fastball Pitch Count, 구종별 투구 수 (투심)',
       '4-Seamer Fastball Pitch Count, 구종별 투구 수 (포심)',
       'Cutter Pitch Count, 구종별 투구 수 (커터)', 'Curve Pitch Count, 구종별 투구 수 (커브)',
       'Slider Pitch Count, 구종별 투구 수 (슬라이더)',
       'Changeup Pitch Count, 구종별 투구 수 (체인지업)',
       'Sinker Pitch Count, 구종별 투구 수 (싱커)',
       'Forkball Pitch Count, 구종별 투구 수 (포크볼)']
        balltype_speed_list = ['2-Seamer Fastball Velocity, 평균구속 (투심)','4-Seamer Fastball Velocity, 평균구속 (포심)',
                              'Cutter Velocity, 평균구속 (커터)', 'Curve Velocity, 평균구속 (커브)',
                               'Slider Velocity, 평균구속 (슬라이더)','Changeup Velocity, 평균구속 (체인지업)',
                              'Sinker Velocity, 평균구속 (싱커)','Forkball Velocity, 평균구속 (포크볼)']
        balltype_100_list = ['2-Seamer Fastball Pitch Value per 100, 구종가치/100 (투심)',
       '4-Seamer Fastball Pitch Value per 100, 구종가치/100 (포심)',
       'Cutter Pitch Value per 100, 구종가치/100 (커터)',
       'Curve Pitch Value per 100, 구종가치/100 (커브)',
       'Slider Pitch Value per 100, 구종가치/100 (슬라이더)',
       'Changeup Pitch Value per 100, 구종가치/100 (체인지업)',
       'Sinker Pitch Value per 100, 구종가치/100 (싱커)',
       'Forkball Pitch Value per 100, 구종가치/100 (포크볼)']
        
    for balltype, balltype_throw, balltype_100 in zip(balltype_list, balltype_throw_list, balltype_100_list):
        balltype_value_list = [float(value) if value != '' else np.nan for value in data[balltype]]
        balltype_throw_num_list = [float(value) if value != '' else np.nan for value in data[balltype_throw]]
        balltype_throw_num_list = np.array(balltype_throw_num_list)
        balltype_throw_num_list[balltype_throw_num_list < 10] = np.nan
        
        data[balltype_100] = balltype_value_list / balltype_throw_num_list * 100
    
    if type == 'batting':
        data = data[['Hitter Number','타자 이름'] + balltype_100_list]
    elif type == 'pitching':
        for balltype_speed in balltype_speed_list:
            data[balltype_speed] = [float(value) if value != '' else np.nan for value in data[balltype_speed]]
        
        data = data[['Pitcher Number','투수 이름'] + balltype_speed_list + balltype_100_list]
    
    return data


def split_finder(player_num,type):
    url = 'http://statiz.sporki.com/player'

    params = {
        'm':'playerinfo',
        'p_no':player_num
    }

    resp = request('GET',url=url, params=params)
    dom = BeautifulSoup(resp.text,'html.parser')
    
    print(player_num)
    
    text = dom.find_all(attrs={'class':'con'})[0].text[-5:-1]
    
    if type == 'pitching':
        pitching_type = text[0:2]
        
        if pitching_type == '우투':
            return 0.0
        
        elif pitching_type == '좌투':
            return 1.0
        
        elif pitching_type == '우언':
            return 0.25
        
        elif pitching_type == '좌언':
            return 0.75
        
    elif type == 'batting':
        batting_type = text[2:4]
        
        if batting_type == '우타':
            return 0.0
        
        elif batting_type == '좌타':
            return 1.0
        
        elif batting_type == '양타':
            return 0.5
        

def gamelog_agg(year):
    gamelog = gamelog_scraper(year)
    batting_player_data = bat_recode(year)
    pitching_player_data = pitch_recode(year)

    # 2024년 4월 24일 현재 전체 시즌 파크 팩터
    park_factors = {'잠실':895, '사직':1064,'창원':1051,'대구':1111,'수원':1020,'문학':985, '고척':965, '대전':1019, '광주':1000,
                '울산':1000,'포항':1000}
    park_factors_list = []
    for stadium in list(gamelog['경기장']):
        park_factors_list.append(park_factors[stadium])
        
    gamelog['파크팩터'] = park_factors_list

    gamelog_agg = pd.merge(gamelog, batting_player_data)
    gamelog_agg = pd.merge(gamelog_agg,pitching_player_data)

    result_list = []

    for result in list(gamelog_agg['결과']):
        if '플라이 아웃' in result:
            result_list.append('아웃')
        elif '땅볼 아웃' in result:
            result_list.append('아웃')
        elif '병살타' in result:
            result_list.append('더블플레이')
        elif '실책' in result:
            result_list.append('1루타')
        elif '1루타' in result:
            result_list.append('1루타')
        elif '2루타' in result:
            result_list.append('2루타')
        elif '3루타' in result:
            result_list.append('3루타')
        elif '내야 안타' in result:
            result_list.append('1루타')
        elif '홈런' in result:
            result_list.append('홈런')
        elif '직선타 아웃' in result:
            result_list.append('아웃')
        elif '삼진' in result:
            result_list.append('아웃')
        elif '야수선택' in result:
            result_list.append('아웃')
        elif '땅볼 출루' in result:
            result_list.append('아웃')
        elif '사구' in result:
            result_list.append('사사구')
        elif '3번트 아웃' in result:
            result_list.append('아웃')
        elif '4구' in result:
            result_list.append('사사구')
        elif '선행주자아웃 출루' in result:
            result_list.append('아웃')
        else:
            result_list.append(None)

    gamelog_agg['Result'] = result_list

    gamelog_agg = gamelog_agg[['날짜','Pitcher Number', 'Hitter Number','투수 이름','타자 이름','내야 수비 RAA',
       '외야 수비 RAA', '포수 수비 RAA', '구종가치/100 (투심)',
       '구종가치/100 (포심)', '구종가치/100 (커터)', '구종가치/100 (커브)', '구종가치/100 (슬라이더)',
       '구종가치/100 (체인지업)', '구종가치/100 (싱커)', '구종가치/100 (포크볼)',
       'Strike%, 전체 투구 대비 스트라이크', 'Called Strike%, 전체 투구 대비 루킹 스트라이크%',
       'Whiff%, 전체 투구 대비 헛스윙 스트라이크%', 'CSW%, 전체 투구 대비 루킹+헛스윙 스트라이크%',
       'Swing%, 스윙 비율', '스윙 대비 콘택트 비율', '스윙 대비 헛스윙 비율', '초구 스트라이크 비율',
       '초구 스윙 비율', '투스트라이크 카운트 투구 대비 삼진 결정 비율', 'Strike Zone%, 존 안에 들어온 투구 비율',
       'Strike Zone Swing%, 존 안에 들어온 투구 대비 스윙 비율',
       'Strike Zone Contact%, 존 안에 들어온 투구 대비 콘택트 비율',
       'Out Zone%, 존 밖에 들어온 투구 비율', 'Out Zone Swing%, 존 밖에 들어온 투구 대비 스윙 비율',
       'Out Zone Contact%, 존 밖에 들어온 투구 대비 콘택트 비율',
       'Meatball Zone%, 존 한가운데 들어온 투구 비율',
       'Meatball Swing%, 존 한가운데 들어온 투구 대비 스윙 비율',
       'Shadow Zone%, 쉐도우존에 들어온 투구 비율', 'Looking Strike Out%, 루킹 삼진 비율',
       'Batting Average on Balls In Play, 인플레이 타구의 안타 비율', '땅볼%', '내야 뜬볼%',
       '외야 뜬볼%', '뜬볼%', '라인드라이브%', '홈런 / 뜬볼%', '내야안타%', '전체 좌측 타구 비율',
       '전체 좌중앙 타구 비율', '전체 중앙 타구 비율', '전체 우중앙 타구 비율', '전체 우측 타구 비율', '투타',
       '(P) Left ball%, 전체 좌측 타구 비율', '(P) Left center ball%, 전체 좌중앙 타구 비율',
       '(P) Center ball%, 전체 중앙 타구 비율', '(P) Right center ball%, 전체 우중앙 타구 비율',
       '(P) Right ball%, 전체 우측 타구 비율', '(P) Pull-side ball%, 전체 당겨친 타구 비율',
        '2-Seamer Fastball Velocity, 평균구속 (투심)',
       '4-Seamer Fastball Velocity, 평균구속 (포심)', 'Cutter Velocity, 평균구속 (커터)',
       'Curve Velocity, 평균구속 (커브)', 'Slider Velocity, 평균구속 (슬라이더)',
       'Changeup Velocity, 평균구속 (체인지업)', 'Sinker Velocity, 평균구속 (싱커)',
       'Forkball Velocity, 평균구속 (포크볼)',
       '2-Seamer Fastball Pitch Value per 100, 구종가치/100 (투심)',
       '4-Seamer Fastball Pitch Value per 100, 구종가치/100 (포심)',
       'Cutter Pitch Value per 100, 구종가치/100 (커터)',
       'Curve Pitch Value per 100, 구종가치/100 (커브)',
       'Slider Pitch Value per 100, 구종가치/100 (슬라이더)',
       'Changeup Pitch Value per 100, 구종가치/100 (체인지업)',
       'Sinker Pitch Value per 100, 구종가치/100 (싱커)',
       'Forkball Pitch Value per 100, 구종가치/100 (포크볼)',
       '(P) Strike%, 전체 투구 대비 스트라이크', '(P) Called Strike%, 전체 투구 대비 루킹 스트라이크%',
       '(P) Whiff%, 전체 투구 대비 헛스윙 스트라이크%', '(P) CSW%, 전체 투구 대비 루킹+헛스윙 스트라이크%',
       '(P) Swing%, 스윙 비율', '(P) 스윙 대비 콘택트 비율', '(P) 스윙 대비 헛스윙 비율',
       '(P) 초구 스트라이크 비율', '(P) 초구 스윙 비율', '(P) 투스트라이크 카운트 투구 대비 삼진 결정 비율',
       '(P) Strike Zone%, 존 안에 들어온 투구 비율',
       '(P) Strike Zone Swing%, 존 안에 들어온 투구 대비 스윙 비율',
       '(P) Strike Zone Contact%, 존 안에 들어온 투구 대비 콘택트 비율',
       '(P) Out Zone%, 존 밖에 들어온 투구 비율',
       '(P) Out Zone Swing%, 존 밖에 들어온 투구 대비 스윙 비율',
       '(P) Out Zone Contact%, 존 밖에 들어온 투구 대비 콘택트 비율',
       '(P) Meatball Zone%, 존 한가운데 들어온 투구 비율',
       '(P) Meatball Swing%, 존 한가운데 들어온 투구 대비 스윙 비율',
       '(P) Looking Strike Out%, 루킹 삼진 비율', '(P) 투타', '파크팩터', 'Result']]
    
    # 1루타(1베이스 출루), 2루타(2베이스 출루), 3루타(3베이스 출루), 홈런, 아웃(삼진 + 플라이 아웃), 더블플레이, 사사구 외 상황 제거
    gamelog_agg = gamelog_agg.dropna(subset=['Result'])

    for col in range(gamelog_agg.shape[1]):
        column_data = gamelog_agg[:, col]
        valid_data = column_data[~np.isnan(column_data)]
        Q1 = np.quantile(valid_data, 0.25)
        Q3 = np.quantile(valid_data, 0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치를 제외한 최솟값 계산
        filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
        filtered_min = np.min(filtered_data)

        # 결측치 대체 전략 설정
        replacement_value = filtered_min if filtered_min < Q3 else Q3

        # 결측치 대체
        gamelog_agg[np.isnan(gamelog_agg[:, col]), col] = replacement_value

    gamelog_agg.to_excel('gamelog_agg_' + str(year) + '.xlsx')

    return gamelog_agg


if __name__ == "__main__":
    # 분석을 원하는 연도 입력
    years = [2022,2023]
    gamelog_all = pd.DataFrame()

    for year in years:
        gamelog = gamelog_agg(year)
        gamelog_all = pd.concat([gamelog_all,gamelog],axis=0)
    
    gamelog_all.to_excel('gamelog_agg.xlsx')
