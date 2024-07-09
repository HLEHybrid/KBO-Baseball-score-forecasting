from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import numpy as np
import preprocessor
import scraper
from pickle import load
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredText
import math

# 마르코프 체인에서 상태 ID를 반환하는 함수
def getID(first, second, third, outs, inning):
    """
    주어진 파라미터로 상태 ID를 반환합니다.
    :param first: 1루에 주자가 있는지 여부 (0 또는 1)
    :param second: 2루에 주자가 있는지 여부 (0 또는 1)
    :param third: 3루에 주자가 있는지 여부 (0 또는 1)
    :param outs: 아웃 수 (0, 1, 2)
    :param inning: 이닝 수 (1-9)
    :returns: int. 주어진 파라미터로 계산된 상태 ID
    """
    return first + 2 * second + 4 * third + 8 * outs + 24 * (inning - 1)

# 마르코프 체인의 상태를 나타내는 클래스
class State:
    """
    마르코프 체인에서 상태를 나타내는 클래스입니다.
    """
    def __init__(self, stateID):
        self.id = stateID
        if stateID == 216:
            self.i = 9
            self.o = 3
            self.t = 0
            self.s = 0
            self.f = 0
        else:  
            self.i = (stateID // 24) + 1
            stateID -= (self.i - 1) * 24
            self.o = stateID // 8
            stateID -= self.o * 8
            self.t = stateID // 4
            stateID -= self.t * 4
            self.s = stateID // 2
            stateID -= self.s * 2
            self.f = stateID

    # 주자가 진루하는 상황들에 대한 함수들
    def walk(self):
        """
        타자가 걸어 나가는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.f == 1:
            if self.s == 1:
                if self.t == 1:
                    return (getID(1, 1, 1, self.o, self.i), 1)
                else:
                    return (getID(1, 1, 1, self.o, self.i), 0)
            else:
                return (getID(1, 1, self.t, self.o, self.i), 0)
        else:
            return (getID(1, self.s, self.t, self.o, self.i), 0)

    def single(self):
        """
        타자가 단타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(1, self.f, self.s, self.o, self.i), self.t)

    def double(self):
        """
        타자가 2루타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 1, self.f, self.o, self.i), self.s + self.t)

    def triple(self):
        """
        타자가 3루타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 0, 1, self.o, self.i), self.f + self.s + self.t)

    def homeRun(self):
        """
        타자가 홈런을 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 0, 0, self.o, self.i), 1 + self.f + self.s + self.t)

    def out(self):
        """
        타자가 아웃되는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.o == 2:
            return (getID(0, 0, 0, 0, self.i + 1), 0)
        else:
            return (getID(self.f, self.s, self.t, self.o + 1, self.i), 0)
    
    def doublePlay(self):
        """
        타자가 병살타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.o >= 1:
            return (getID(0, 0, 0, 0, self.i + 1), 0)
        else:
            return (getID(self.f, self.s, self.t, self.o + 2, self.i), 0)

# 야구 선수 정보를 나타내는 클래스
class Player:
    """
    야구 선수를 나타내는 클래스입니다.
    """
    def __init__(self, playerID, name, first, second, third, bb, homerun, outs, double):
        """
        :param playerID: int. 선수의 고유 ID.
        :param name: string. 선수의 이름.
        :param first: float. 단타 확률.
        :param second: float. 2루타 확률.
        :param third: float. 3루타 확률.
        :param bb: float. 볼넷 확률.
        :param homerun: float. 홈런 확률.
        :param outs: float. 아웃 확률.
        :param double: float. 병살 확률.
        """
        self.id = playerID
        self.name = name
        self.first = first
        self.second = second
        self.third = third
        self.double = double
        self.bb = bb
        self.outs = outs
        self.homerun = homerun

    def transitionMatrixSimple(self):
        """
        이 선수에 대한 전이 행렬을 계산합니다.
        :return: numpy (217, 217) 배열. 이 선수의 전이 행렬.
        """
        p = np.zeros((5, 217, 217))
        p[0][216][216] = 1

        for i in range(216):
            currState = State(i)
            nextState, runs = currState.walk()
            p[runs][i][nextState] += self.bb
            nextState, runs = currState.single()
            p[runs][i][nextState] += self.first
            nextState, runs = currState.double()
            p[runs][i][nextState] += self.second
            nextState, runs = currState.triple()
            p[runs][i][nextState] += self.third
            nextState, runs = currState.homeRun()
            p[runs][i][nextState] += self.homerun
            nextState, runs = currState.out()
            p[runs][i][nextState] += self.outs
            nextState, runs = currState.doublePlay()
            p[runs][i][nextState] += self.double
        return p
    
def expectedRuns(lineup):
    """
    주어진 야구 라인업의 예상 득점 분포를 계산합니다.
    :param lineup: [Batter]. 라인업에 포함된 9명의 타자 리스트.
    :return: np.array. 21개의 요소를 포함하는 배열. i번째 요소는 라인업이 i 득점할 확률을 나타냅니다.
    """
    transitionsMatrices = list(map(lambda Batter: Batter.transitionMatrixSimple(), lineup))
    return simulateMarkovChain(transitionsMatrices)[:, 216]

def simulateMarkovChain(transitionMatrices):
    """
    야구 게임을 나타내는 마르코프 체인의 거의 정적 상태 분포를 찾습니다.
    :param transitionMatrices: [numpy array]. 라인업에 포함된 타자에 대한 9개의 (217x217) 전이 행렬 리스트.
    :return: numpy 21x217 배열. 배열의 i번째 행은 i 득점이 된 상태를 나타냅니다.
    """
    u = np.zeros((21, 217))
    u[0][0] = 1
    iterations = 0
    batter = 0
    while sum(u)[216] < 0.999 and iterations < 2000:
        p = transitionMatrices[batter]
        next_u = np.zeros((21, 217))
        for i in range(21):
            for j in range(5):
                if i - j >= 0:
                    next_u[i] += u[i-j] @ p[j]
        u = next_u
        batter = (batter + 1) % 9 
        iterations += 1
    return u

def teamExpectedRuns(teamName, opponent_team_name, starter_lineup_list, relief_lineup_list, starter_data, starter_name, starter_num):
    """
    주어진 팀의 예상 득점을 계산하고 결과를 출력합니다.
    :param teamName: 팀 이름.
    :param starter_lineup_list: 선발 투수 라인업 리스트.
    :param relief_lineup_list: 구원 투수 라인업 리스트.
    :param starter_data: 선발 투수 데이터.
    :param starter_num: 선발 투수 번호.
    :param opponent_name: 상대팀 이름
    """
    print('\n팀: ' + teamName + '\n')
    print('상대팀: ' + opponent_team_name + '\n')
    print('상대 선발 투수: ' + starter_name + '\n')
    print('라인업: ' + str(list(map(lambda Batter: Batter.name, starter_lineup_list[0]))) + '\n')
    
    try:
        avg_inning = float(starter_data[starter_data['Number'] == starter_num].loc[:, 'IP per GS, 선발 경기당 이닝 수'])
        if avg_inning < 5.0:
            avg_inning = 5.0
    except:
        avg_inning = 5.0
    
    # 선발 투수 득점 계산
    u = expectedRuns(starter_lineup_list[0])
    starter_expRuns = 0
    if sum(u) < 0.7:
        print('게임 종료 확률이 낮아 예상 실점을 계산할 수 없습니다.')
        u = (1/sum(u))*u
        
        for i in range(21):
            starter_expRuns += i * u[i]
        
        avg_inning = 9 * (4/starter_expRuns)
    else:
        for i in range(21):
            starter_expRuns += i * u[i]
        if (avg_inning / 9) * starter_expRuns > 4:
            avg_inning = 9 * (4/starter_expRuns)
        
    # 불펜 투수 득점 계산
    

    relief_exp_runs_list = []
    for relief_lineup in relief_lineup_list:
        exp = expectedRemainingRuns(relief_lineup, 0, State(getID(0, 0, 0, 0, 1)))
        relief_exp_runs_list.append(exp)
        
    relief_expRuns = sum(relief_exp_runs_list) / len(relief_exp_runs_list)
    total_expRuns = (avg_inning / 9) * starter_expRuns + ((9 - avg_inning) / 9) * relief_expRuns

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(u)), u, color='blue')
    ax.set_xlabel('Runs Scored')
    ax.set_ylabel('Probability')
    ax.set_title(f'{teamName}의 선발 상대 예상 득점 분포')
    ax.legend()

    # 예상 득점 텍스트 추가
    anchored_text = AnchoredText(
        f'상대 팀: {opponent_team_name}\nStarter Expected Runs: {(avg_inning / 9) * starter_expRuns:.2f}\nRelief Expected Runs: {((9 - avg_inning) / 9) * relief_expRuns:.2f}\nTotal Expected Runs: {total_expRuns:.2f}',
        loc='upper right', prop=dict(size=10)
    )
    ax.add_artist(anchored_text)
    
    plt.savefig(f'img/{teamName}.png')

    print('게임 종료 확률: ' + str(sum(u)) + '\n')
    print('\n선발 투수 평균 이닝 수: ' + str(avg_inning) + '\n')
    print('선발 투수에 의한 예상 실점: ' + str((avg_inning / 9) * starter_expRuns) + '\n')
    print('선발 투수에 의한 예상 실점(9이닝 당): ' + str(starter_expRuns) + '\n')

    print('각 점수에 대한 확률:')
    for i in range(21):
        print(str(i) + ': ' + str(u[i]))
    
    print('\n구원 투수 예상 실점: ' + str(((9 - avg_inning) / 9) * relief_expRuns) + '\n')

    print('\n총 예상 실점:' + str(total_expRuns) + '\n')
    
    return total_expRuns


def expectedRemainingRuns(lineup, batterUp, startState):
    """
    게임의 특정 지점에서 팀이 득점할 예상 점수를 계산합니다.
    :param lineup: 9명의 타자 리스트.
    :param batterUp: 타석에 있는 타자의 인덱스 (0-8).
    :param startState: 현재 게임의 상태.
    :return: 주어진 상태에서 팀이 득점할 예상 점수.
    """
    transitionsMatrices = list(map(lambda Batter: Batter.transitionMatrixSimple(), lineup))
    u = np.zeros((21, 217))
    u[0][startState.id] = 1
    iterations = 0
    batter = batterUp
    while sum(u)[216] < 0.999 and iterations < 2000:
        p = transitionsMatrices[batter]
        next_u = np.zeros((21, 217))
        for i in range(21):
            for j in range(5):
                if i - j >= 0:
                    next_u[i] += u[i-j] @ p[j]
        u = next_u
        batter = (batter + 1) % 9 
        iterations += 1
    u = u[:, 216]
    expRuns = 0
    if sum(u) < 0.7:
        # u = (1/sum(u))*u
        expRuns = 9
    else:
        for i in range(21):
            expRuns += i * u[i]
        if expRuns > 9:
            expRuns = 9
    return expRuns

def split_changer(split_name, position):
    """
    투타 정보를 숫자로 변환합니다.
    :param split_name: 투타 정보 (우투, 좌투 등).
    :param position: 포지션 (투수 또는 타자).
    :return: float. 변환된 숫자 값.
    """
    split_num = 0.0
    if position == 'pitching':
        if split_name == '우투' or split_name == '우완투수':
            split_num = 0.0
        elif split_name == '좌투' or split_name == '좌완투수':
            split_num = 1.0
        elif split_name == '우언' or split_name == '우완언더':
            split_num = 0.25
        elif split_name == '좌언' or split_name == '좌완언더':
            split_num = 0.75
        
    elif position == 'batting':
        if split_name == '우타':
            split_num = 0.0
        elif split_name == '좌타':
            split_num = 1.0
        elif split_name == '양타':
            split_num = 0.5
            
    return split_num

def pitcher_batter_aug(pitcher_data, batter_data, defence, outfield_recode, infield_recode, catcher_recode, stadium):
    """
    투수와 타자 데이터를 결합하여 필요한 데이터를 생성합니다.
    :param pitcher_data: 투수 데이터.
    :param batter_data: 타자 데이터.
    :param defence: 수비 데이터.
    :param outfield_recode: 외야 수비 데이터.
    :param infield_recode: 내야 수비 데이터.
    :param catcher_recode: 포수 수비 데이터.
    :param stadium: 경기장 정보.
    :return: 결합된 데이터 리스트.
    """
    batter_pitcher_list = []
    outfield_raa_sum, infield_raa_sum, catcher_raa_sum = scraper.calculate_raa_sums(defence, outfield_recode, infield_recode, catcher_recode)
    for pitcher_num in list(pitcher_data['Pitcher Number']):
        batter_pitcher = pd.DataFrame()
        for player_num in list(batter_data['Hitter Number']):
            row = pd.DataFrame({
                'Pitcher Number': [pitcher_num],
                'Hitter Number': [player_num],
                '내야 수비 RAA': [outfield_raa_sum],
                '외야 수비 RAA': [infield_raa_sum],
                '포수 수비 RAA': [catcher_raa_sum]
            })
            batter_pitcher = pd.concat([batter_pitcher, row], ignore_index=True)
        batter_pitcher = pd.merge(batter_pitcher,pitcher_data)
        batter_pitcher = pd.merge(batter_pitcher,batter_data)
        park_factors = {'잠실':895, '사직':1064,'창원':1051,'대구':1111,'수원':1020,'문학':985, '고척':965, '대전':1019, '광주':1000,
                    '울산':1000,'포항':1000,'청주':1000}
        batter_pitcher['파크팩터'] = park_factors[stadium]
        
        columns_to_select = [
            'Pitcher Number', 'Hitter Number','투수 이름','타자 이름','내야 수비 RAA',
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
            '(P) Looking Strike Out%, 루킹 삼진 비율', '(P) 투타', '파크팩터']
        batter_pitcher = batter_pitcher[columns_to_select]
        batter_pitcher_list.append(batter_pitcher)
        
    return batter_pitcher_list

def make_prob_lineup(data_list, load_scaler, model):
    """
    타자와 투수 데이터를 사용하여 확률적 라인업을 생성합니다.
    :param data_list: 결합된 데이터 리스트.
    :param load_scaler: 스케일러 객체.
    :param model: 예측 모델.
    :return: 확률적 라인업 리스트.
    """
    lineup_list = []
    for data in data_list:
        x = data.iloc[:,4:].values
        x = load_scaler.transform(x)
        y_predict = model.predict(x)
        
        lineup = []
        playerIDs = list(data.iloc[:,1])
        playerNames = list(data.iloc[:,3])
        
        for i in range(9):
            playerID = playerIDs[i]
            name = playerNames[i]
            first = y_predict[i][0]
            second = y_predict[i][1]
            third = y_predict[i][2]
            double = y_predict[i][3]
            bb = y_predict[i][4]
            outs = y_predict[i][5]
            homerun = y_predict[i][6]
            lineup.append(Player(playerID, name, first, second, third, double, bb, outs, homerun))
            
        lineup_list.append(lineup)
        
    return lineup_list

def today_lineup(bat_recode, pitch_recode, year, date):
    """
    오늘의 경기 라인업을 가져오고 예상 득점을 계산합니다.
    :param bat_recode: 타자 기록 데이터.
    :param pitch_recode: 투수 기록 데이터.
    :param year: 년도.
    :param date: 날짜 (YYYY-MM-DD).
    """
    for col in pitch_recode.columns[2:]:
        pitch_recode[col] = pd.to_numeric(pitch_recode[col], errors='coerce')
    
    for col in bat_recode.columns[2:]:
        bat_recode[col] = pd.to_numeric(bat_recode[col], errors='coerce')    
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 브라우저 창을 열지 않음
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)
    url = f'https://m.sports.naver.com/kbaseball/schedule/index?date={date}'
    driver.get(url)
    time.sleep(3)  # 페이지가 로드될 때까지 잠시 대기
    
    today_schedule = driver.find_elements(By.CLASS_NAME, 'ScheduleAllType_match_list__3n5L_')
    matchs = today_schedule[0].find_elements(By.CLASS_NAME, 'MatchBoxLinkArea_link_match__3MbV_')
    match_link_list = list()

    for i in range(int(math.ceil(len(matchs) / 2))):
        match_link = matchs[2 * i].get_attribute('href')
        match_link = match_link.replace('preview', 'lineup')
        match_link = match_link.replace('video', 'lineup')
        match_link = match_link.replace('relay', 'lineup')
        match_link = match_link.replace('record', 'lineup')
        match_link = match_link.replace('cheer', 'lineup')
        match_link_list.append(match_link)

    results = {}
    for link in match_link_list:
        try:
            driver.get(link)
            time.sleep(3)  # 페이지가 로드될 때까지 잠시 대기
            print(link)
            
            lineup_raw_data = driver.find_elements(By.CLASS_NAME, 'Lineup_lineup_list__1_CNQ')
            backup_player_raw_data = driver.find_elements(By.CLASS_NAME, 'Lineup_player_area__w8egy')
            stadium = driver.find_elements(By.CLASS_NAME, 'MatchBox_stadium__17mQ4')[0].text
            
            away_lineup_data = lineup_raw_data[0].text.split('\n')
            home_lineup_data = lineup_raw_data[1].text.split('\n')
            
            away_backup_fielder_data = backup_player_raw_data[0].text.split('\n')
            home_backup_fielder_data = backup_player_raw_data[1].text.split('\n')
            away_backup_pitcher_data = backup_player_raw_data[2].text.split('\n')
            home_backup_pitcher_data = backup_player_raw_data[3].text.split('\n')
            
            home_start_pitcher = home_lineup_data[1]
            home_start_pitcher_split = split_changer(home_lineup_data[2], 'pitching')
            away_start_pitcher = away_lineup_data[1]
            away_start_pitcher_split = split_changer(away_lineup_data[2], 'pitching')
            
            try:
                home_start_pitcher_num = pitch_recode[(pitch_recode['투수 이름'] == home_start_pitcher) & (pitch_recode['(P) 투타'] == home_start_pitcher_split)].iloc[0, 0]
                home_start_pitcher_data = pitch_recode[pitch_recode['Pitcher Number'] == home_start_pitcher_num]
            except:
                # 모든 열의 평균 계산
                numeric_means = pitch_recode.select_dtypes(include='number').mean()

                # 비숫자형 열의 첫 번째 값 가져오기
                non_numeric_data = pitch_recode.select_dtypes(exclude='number').iloc[0]

                # 평균값과 비숫자형 데이터를 결합
                mean_data = pd.concat([numeric_means, non_numeric_data])

                # 평균값을 데이터프레임으로 변환
                home_start_pitcher_num = 00000
                home_start_pitcher_data = pd.DataFrame(mean_data).transpose()
                home_start_pitcher_data['Pitcher Number'] = 00000
                home_start_pitcher_data['투수 이름'] = home_start_pitcher
                home_start_pitcher_data['(P) 투타'] = home_start_pitcher_split
                
            try:
                away_start_pitcher_num = pitch_recode[(pitch_recode['투수 이름'] == away_start_pitcher) & (pitch_recode['(P) 투타'] == away_start_pitcher_split)].iloc[0, 0]
                away_start_pitcher_data = pitch_recode[pitch_recode['Pitcher Number'] == away_start_pitcher_num]
            except:
                # 모든 열의 평균 계산
                numeric_means = pitch_recode.select_dtypes(include='number').mean()

                # 비숫자형 열의 첫 번째 값 가져오기
                non_numeric_data = pitch_recode.select_dtypes(exclude='number').iloc[0]

                # 평균값과 비숫자형 데이터를 결합
                mean_data = pd.concat([numeric_means, non_numeric_data])

                # 평균값을 데이터프레임으로 변환
                away_start_pitcher_num = 00000
                away_start_pitcher_data = pd.DataFrame(mean_data).transpose()
                away_start_pitcher_data['Pitcher Number'] = 00000
                away_start_pitcher_data['투수 이름'] = away_start_pitcher
                away_start_pitcher_data['(P) 투타'] = away_start_pitcher_split
            
            home_batters_data = pd.DataFrame()
            away_batters_data = pd.DataFrame()
            
            home_defence = {}
            away_defence = {}
            
            for i in range(9):
                home_batter = home_lineup_data[3 * i + 4]
                home_position = home_lineup_data[3 * i + 5].split(' , ')[0]
                home_split = split_changer(home_lineup_data[3 * i + 5].split(' , ')[1], 'batting')
                        
                try:
                    home_batter_num = bat_recode[(bat_recode['타자 이름'] == home_batter) & (bat_recode['투타'] == home_split)].iloc[0, 0]
                    home_defence[home_position] = home_batter_num
                    batter_data = bat_recode[bat_recode['Hitter Number'] == home_batter_num]
                    home_batters_data = pd.concat([home_batters_data, batter_data], ignore_index=True)
                        
                except:
                    # 모든 열의 평균 계산
                    numeric_means = bat_recode.select_dtypes(include='number').mean()

                    # 비숫자형 열의 첫 번째 값 가져오기
                    non_numeric_data = bat_recode.select_dtypes(exclude='number').iloc[0]

                    # 평균값과 비숫자형 데이터를 결합
                    mean_data = pd.concat([numeric_means, non_numeric_data])

                    # 평균값을 데이터프레임으로 변환
                    home_defence[home_position] = 99999 - i
                    batter_data = pd.DataFrame(mean_data).transpose()
                    batter_data['Hitter Number'] = 99999 - i
                    batter_data['타자 이름'] = home_batter
                    batter_data['투타'] = home_split  
                    home_batters_data = pd.concat([home_batters_data, batter_data], ignore_index=True)

                away_batter = away_lineup_data[3 * i + 4]
                away_position = away_lineup_data[3 * i + 5].split(' , ')[0]
                away_split = split_changer(away_lineup_data[3 * i + 5].split(' , ')[1], 'batting')
                        
                try:
                    away_batter_num = bat_recode[(bat_recode['타자 이름'] == away_batter) & (bat_recode['투타'] == away_split)].iloc[0, 0]
                    away_defence[away_position] = away_batter_num
                    batter_data = bat_recode[bat_recode['Hitter Number'] == away_batter_num]
                    away_batters_data = pd.concat([away_batters_data, batter_data], ignore_index=True)
                        
                except:
                    # 모든 열의 평균 계산
                    numeric_means = bat_recode.select_dtypes(include='number').mean()

                    # 비숫자형 열의 첫 번째 값 가져오기
                    non_numeric_data = bat_recode.select_dtypes(exclude='number').iloc[0]

                    # 평균값과 비숫자형 데이터를 결합
                    mean_data = pd.concat([numeric_means, non_numeric_data])

                    # 평균값을 데이터프레임으로 변환
                    away_defence[away_position] = 99999 - i
                    batter_data = pd.DataFrame(mean_data).transpose()
                    batter_data['Hitter Number'] = 99999 - i
                    batter_data['타자 이름'] = away_batter
                    batter_data['투타'] = away_split
                    away_batters_data = pd.concat([away_batters_data, batter_data], ignore_index=True)
            
            home_team_name = home_backup_pitcher_data[0].split(' ')[0]
            away_team_name = away_backup_pitcher_data[0].split(' ')[0]
            
            # 불펜 투수 계산
            
            home_relief_data = pd.DataFrame()
            away_relief_data = pd.DataFrame()
            
            starter_data = scraper.recode_scraper('pitching', 'starting', year)
            
            for col in starter_data.columns[3:]:
                starter_data[col] = pd.to_numeric(starter_data[col], errors='coerce')
            starter_data = starter_data[starter_data['Games started, 선발 등판 횟수'] / starter_data['Games, 출장'] > 0.5]
            
            for i in range(int((len(home_backup_pitcher_data) - 1) / 2)):
                pitcher_name = home_backup_pitcher_data[2 * i + 1]
                pitcher_split = split_changer(home_backup_pitcher_data[2 * i + 2], 'pitching')
                try:
                    pitcher_num = pitch_recode[(pitch_recode['투수 이름'] == pitcher_name) & (pitch_recode['(P) 투타'] == pitcher_split)].iloc[0, 0]
                    if pitcher_num not in list(starter_data['Number']):
                        pitcher_data = pitch_recode[pitch_recode['Pitcher Number'] == pitcher_num]
                        home_relief_data = pd.concat([home_relief_data, pitcher_data], ignore_index=True)
                except:
                    continue
                    
            for i in range(int((len(away_backup_pitcher_data) - 1) / 2)):
                pitcher_name = away_backup_pitcher_data[2 * i + 1]
                pitcher_split = split_changer(away_backup_pitcher_data[2 * i + 2], 'pitching')
                try:
                    pitcher_num = pitch_recode[(pitch_recode['투수 이름'] == pitcher_name) & (pitch_recode['(P) 투타'] == pitcher_split)].iloc[0, 0]
                    if pitcher_num not in list(starter_data['Number']):
                        pitcher_data = pitch_recode[pitch_recode['Pitcher Number'] == pitcher_num]
                        away_relief_data = pd.concat([away_relief_data, pitcher_data], ignore_index=True)
                except:
                    continue

            # 수비 지표 추가
            outfield_recode = scraper.recode_scraper('fielding', 'outField', year)
            infield_recode = scraper.recode_scraper('fielding', 'inField', year)
            catcher_recode = scraper.recode_scraper('fielding', 'catcher', year)
            
            home_batter_away_starter_list = pitcher_batter_aug(away_start_pitcher_data, home_batters_data, away_defence, outfield_recode, infield_recode, catcher_recode, stadium)
            home_batter_away_relief_list = pitcher_batter_aug(away_relief_data, home_batters_data, away_defence, outfield_recode, infield_recode, catcher_recode, stadium)
            away_batter_home_starter_list = pitcher_batter_aug(home_start_pitcher_data, away_batters_data, home_defence, outfield_recode, infield_recode, catcher_recode, stadium)
            away_batter_home_relief_list = pitcher_batter_aug(home_relief_data, away_batters_data, home_defence, outfield_recode, infield_recode, catcher_recode, stadium)

            load_scaler = load(open('scaler.pkl', 'rb'))

            model_path = 'models/kbo_model_dnn.hdf5'
            model = tf.keras.models.load_model(model_path) 

            home_batter_away_starter_lineup_list = make_prob_lineup(home_batter_away_starter_list, load_scaler, model)
            home_batter_away_relief_lineup_list = make_prob_lineup(home_batter_away_relief_list, load_scaler, model)
            away_batter_home_starter_lineup_list = make_prob_lineup(away_batter_home_starter_list, load_scaler, model)
            away_batter_home_relief_lineup_list = make_prob_lineup(away_batter_home_relief_list, load_scaler, model)
            
            print('예상 득점 계산 중')
            
            home_expRuns = teamExpectedRuns(home_team_name, away_team_name, home_batter_away_starter_lineup_list, home_batter_away_relief_lineup_list, starter_data, away_start_pitcher, away_start_pitcher_num)
            away_expRuns = teamExpectedRuns(away_team_name, home_team_name, away_batter_home_starter_lineup_list, away_batter_home_relief_lineup_list, starter_data, home_start_pitcher, home_start_pitcher_num)

            # 승률 계산
            exp_runs_diff = home_expRuns - away_expRuns
            home_win_prob = logistic_win_prob(exp_runs_diff)
            away_win_prob = 1 - home_win_prob

            print(f'\n{home_team_name}의 예상 승률: {home_win_prob:.2%}')
            print(f'{away_team_name}의 예상 승률: {away_win_prob:.2%}')

            results[link] = {
                'home_team': home_team_name,
                'away_team': away_team_name,
                'home_expRuns': home_expRuns,
                'away_expRuns': away_expRuns,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob
            }
        except Exception as e:
            print('이 경기는 라인업이 뜨지 않았거나 취소되었습니다.')
            print(e)

    driver.close()
    return results

def logistic_win_prob(exp_runs_diff, alpha=0.2):
    """
    기대 득점 차이를 사용하여 승률을 계산합니다.
    :param exp_runs_diff: 기대 득점 차이 (홈 팀 - 어웨이 팀).
    :param alpha: 로지스틱 함수의 민감도 파라미터 (기본값: 0.2).
    :return: 홈 팀 승률.
    """
    return 1 / (1 + np.exp(-alpha * exp_runs_diff))
    
def save_results_as_image(results, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    text = ""
    for link, result in results.items():
        text += f"\n경기 링크: {link}\n"
        text += f"{result['home_team']} vs {result['away_team']}\n"
        text += f"{result['home_team']} 예상 득점: {result['home_expRuns']}\n"
        text += f"{result['away_team']} 예상 득점: {result['away_expRuns']}\n"
        text += f"{result['home_team']} 승률: {result['home_win_prob']:.2%}\n"
        text += f"{result['away_team']} 승률: {result['away_win_prob']:.2%}\n\n"
        
    anchored_text = AnchoredText(text, loc='upper left', prop=dict(size=10), frameon=True)
    ax.add_artist(anchored_text)
    
    ax.set_axis_off()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    
if __name__ == '__main__':
    # 한글 폰트 설정
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 또는 사용자의 시스템에 설치된 다른 한글 폰트 경로
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())

    bat_recode = preprocessor.bat_recode(2024)
    pitch_recode = preprocessor.pitch_recode(2024)

    # 싱커 데이터 부족으로 인한 임시 방편
    bat_recode['구종가치/100 (싱커)'] = -0.899598784
    pitch_recode['Sinker Velocity, 평균구속 (싱커)'] = 126.1545879
    pitch_recode['Sinker Pitch Value per 100, 구종가치/100 (싱커)'] = -1.485315299

    results = today_lineup(bat_recode, pitch_recode, 2024, '2024-07-09')
    save_results_as_image(results, 'img/results.png')