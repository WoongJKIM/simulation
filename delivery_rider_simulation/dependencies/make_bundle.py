# 점수를 시그모이드로 만듦

import random
import datetime as dt
import time
import math
import itertools

import numpy as np
import pandas as pd

##### x,y 구하는 함수set

def lat_lng_to_coord(lat, lng):
    r = 6371
    lat_1 = 90 - lat
    x = r * math.sin(math.radians(lat_1)) * math.cos(math.radians(lng))
    y = r * math.sin(math.radians(lat_1)) * math.sin(math.radians(lng))
    z = r * math.cos(math.radians(lat_1))

    return np.transpose(np.matrix([[x, y, z, 1]]))


def yaw(o_lat): # 중심이 될 위치의 lat 만큼 역회전
    o_lat_1 = -(90 - o_lat)

    yaw = math.radians(o_lat_1)
    return np.matrix([[math.cos(yaw), 0, math.sin(yaw), 0],[0, 1, 0, 0],[-math.sin(yaw), 0, math.cos(yaw), 0],[0, 0, 0, 1]])

def roll(o_lng): # 중심이 될 위치의 lng 만큼 역회전
    o_lng_1 = - o_lng
    roll = math.radians(o_lng_1)
    return np.matrix([[math.cos(roll), -math.sin(roll), 0, 0],[math.sin(roll), math.cos(roll), 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])

def matrix_rotation(lat,lng,o_lat,o_lng):
    x = np.dot(np.dot(yaw(o_lat), roll(o_lng)), lat_lng_to_coord(lat, lng))[1]
    y = - (np.dot(np.dot(yaw(o_lat), roll(o_lng)), lat_lng_to_coord(lat, lng))[0])

    # 강남 지역의 블록에 맞춰 회전
    x_rotate = x.item() * 3.135473886 / 3.362955078 - y.item() * (- 1.215841425 / 3.362955078)
    y_rotate = x.item() * (-1.215841425 / 3.362955078) + y.item() * 3.135473886 / 3.362955078

    return x_rotate, y_rotate

##### x,y 를 이용한 두 점의 거리 구하는 함수

def u_dist_coord(x1, y1, x2, y2):
    u_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 )
    return u_dist

def m_dist_coord(x1_rotate, y1_rotate, x2_rotate, y2_rotate):
    m_dist = abs(x2_rotate - x1_rotate) + abs(y2_rotate - y1_rotate)
    return m_dist


def set_candidates_list(sample_cnt, df):
    candidate_list = list(df.index)
    candidate_len = len(candidate_list)

    # 샘플 100개 만들기
    candidates_list = []
    t_candidate_len = candidate_len +  math.floor(candidate_len / 2)

    for idx in range(0, sample_cnt):
        random.seed(time.time())

        leng = random.randint(1, candidate_len - 1)
        base = random.randint(0, leng - 1)
        
        candidate_list[base : candidate_len if (base + leng) > candidate_len else (base + leng)] = reversed(candidate_list[base : candidate_len if (base + leng) > candidate_len else (base + leng)])
        
        cand_bun_list = random.sample(list(range(0, t_candidate_len - 2)), random.randint(0, math.floor(candidate_len / 2))) #
        cand_bun_list.sort()
        
        cand_seq_list = candidate_list.copy()
        candidates_list.append([cand_seq_list, cand_bun_list])

    return candidates_list

def set_trans_candidate_list(candidate_list):
    
    candidate_len = len(candidate_list[0])
    t_candidate_len = candidate_len + math.floor(candidate_len / 2)
    
    cand_seq_list = candidate_list[0]
    cand_bun_list = candidate_list[1]

    b_slot = 0
    b_val = 0
    a_slot = len(cand_seq_list)
    trans_candidate_list = []

    for a_slot in cand_bun_list:
        trans_candidate_list.extend(cand_seq_list[b_slot : a_slot - b_val].copy())
        trans_candidate_list.append(-1)

        b_slot = a_slot - b_val
        b_val += 1
    
    trans_candidate_list.extend(cand_seq_list[b_slot : len(cand_seq_list)])
    trans_candidate_list.extend([-1 for _ in range((t_candidate_len) - len(trans_candidate_list))])

    return trans_candidate_list

# def set_degree_weight(x, y, _x, _y):
#     degree_weight = 1
#     if ((x == 0) & (y == 0)) | ((_x == 0) & (_y == 0)):
#         pass
#     else:
#         a = math.sqrt(pow(_x - x, 2) + pow(_y - y, 2))
#         b = math.sqrt(pow(_x, 2) + pow(_y, 2))
#         c = math.sqrt(pow(x, 2) + pow(y, 2))

#         cos_a = (pow(b, 2) + pow(c, 2) - pow(a, 2))/(2 * b * c)
#         degree_weight = 3 - (2 * cos_a)

#     return degree_weight

def _set_degree_weight(coord):
    
    _x, _y, x, y = coord
    degree_weight = 1
    if ((x == 0) & (y == 0)) | ((_x == 0) & (_y == 0)):
        pass
    else:
        a = math.sqrt(pow(_x - x, 2) + pow(_y - y, 2))
        b = math.sqrt(pow(_x, 2) + pow(_y, 2))
        c = math.sqrt(pow(x, 2) + pow(y, 2))

        cos_a = (pow(b, 2) + pow(c, 2) - pow(a, 2))/(2 * b * c)
        degree_weight = 3 - (2 * cos_a)

    return degree_weight

def set_degree_weight(coord):

    degree_weight_list = list(map(_set_degree_weight, coord))
    
    return degree_weight_list


def set_mutate_candidate_list(candidate_list):
    trans_candidate_list = set_trans_candidate_list(candidate_list)
        
    while True:
        sample_list = random.sample(range(len(trans_candidate_list)), 2)
        if trans_candidate_list[sample_list[0]] != trans_candidate_list[sample_list[1]]:
            break

    change_val = 0
    mutation_0_val = trans_candidate_list[sample_list[0]]
    mutation_1_val = trans_candidate_list[sample_list[1]]

    change_val = mutation_0_val
    trans_candidate_list[sample_list[0]] = mutation_1_val
    trans_candidate_list[sample_list[1]] = change_val

    cand_seq_list = []
    cand_bun_list = []

    for jdx in range(len(trans_candidate_list)):

        if trans_candidate_list[jdx] >= 0:
            cand_seq_list.append(trans_candidate_list[jdx])
        else:
            cand_bun_list.append(jdx)

    del change_val, mutation_0_val, mutation_1_val, trans_candidate_list

    return [cand_seq_list, cand_bun_list]

def set_sigmoid(avg_score, std_score, score):
    
    sigmoid_val = 1 / (1 + np.exp(0 if std_score == 0 else (- (avg_score - score) / std_score)))

    return sigmoid_val

# 후보자 점수를 만드는 것을 시그모이드 형태(0, 1 사이 값)로 생성해 합함
def set_cand_score_df(now_at, epoch, candidates_list, real_df):
    
    trans_candidates_list = [set_trans_candidate_list(candidate_list) for candidate_list in candidates_list]
    
    # 후보자들의 점수를 계산함
    cand_score = setCandScore()
    sub_candidates_score_list = []

    [sub_candidates_score_list.append(cand_score.set_candidates_score_list(now_at, candidate_list, real_df)) for candidate_list in trans_candidates_list]
    
    candidates_score_df = pd.DataFrame(sub_candidates_score_list, columns = ['deli_duration', 'delay_duration', '_deli_duration'])
    
    avg_deli_score = candidates_score_df._deli_duration.mean()
    std_deli_score = candidates_score_df._deli_duration.std()
    avg_delay_score = candidates_score_df.delay_duration.mean()
    std_delay_score = candidates_score_df.delay_duration.std()

    candidates_score_df.loc[:, 'deli_score'] = set_sigmoid(avg_deli_score, std_deli_score, candidates_score_df['_deli_duration'].values)
    candidates_score_df.loc[:, 'delay_score'] = set_sigmoid(avg_delay_score, std_delay_score, candidates_score_df['delay_duration'].values)
    candidates_score_df.loc[:, 'score'] = candidates_score_df['deli_score'].values + candidates_score_df['delay_score'].values
    
    del trans_candidates_list, sub_candidates_score_list

    return candidates_score_df

def set_new_candidate_list(epoch, candidates_list, candidates_score_df):

    # 계산한 점수를 가지고 룰렛판을 만듦
    candidates_score_list = list(candidates_score_df.score.values)
    total_score = sum(candidates_score_list)

    roulette = setRoulWheelDF()
    roulette_list = []
    [roulette_list.append(roulette.set_roulette_df(score / total_score)) for score in candidates_score_list]

    roulette_df = pd.DataFrame(list(roulette_list), columns = ['score', 'start_score', 'end_score'])
    
    roul_s_score = roulette_df.start_score.values
    roul_e_score = roulette_df.end_score.values

    new_candidates_list = []

    candidates_len = len(candidates_list)
    cross_ratio = 0.9
    copy_sample_cnt = math.floor(candidates_len * cross_ratio)
    cross_sample_cnt = candidates_len - copy_sample_cnt
    
    # 복사
    #  - 점수가 가장 높음 3개 - 총점이 가장 높음, 라이더 관점 배달 점수 가장 높음, 지연 시간 점수 가장 높음
    best_score_idx = candidates_score_df.sort_values(['score', 'deli_score', 'delay_score'], ascending = [False, False, False]).head(1).index.values[0]
    best_deli_score_idx = candidates_score_df.sort_values(['deli_score', 'delay_score', 'score'], ascending = [False, False, False]).head(1).index.values[0]
    best_delay_score_idx = candidates_score_df.sort_values(['delay_score', 'deli_score', 'score'], ascending = [False, False, False]).head(1).index.values[0]

    new_candidates_list.append(candidates_list[best_score_idx].copy())
    new_candidates_list.append(candidates_list[best_deli_score_idx].copy())
    new_candidates_list.append(candidates_list[best_delay_score_idx].copy())

    #  - 룰렛으로 복사
    picking_list = np.array([random.random() for _ in range(3, copy_sample_cnt)])

    copy_sample_list, j = np.where((roul_s_score[:, None] <= picking_list) & (picking_list < roul_e_score[:, None]))
    [new_candidates_list.append(candidates_list[idx].copy()) for idx in copy_sample_list]

    # 교배
    picking_seq_list = np.array([random.random() for _ in range(cross_sample_cnt)])
    
    j, cross_sample_list = np.where((roul_s_score <= picking_seq_list[:, None]) & (picking_seq_list[:, None] < roul_e_score))
    
    [new_candidates_list.append([candidates_list[cross_sample_list[idx * 2]][0].copy(), candidates_list[cross_sample_list[idx * 2 + 1]][1].copy()]) for idx in range(int(cross_sample_cnt / 2))]
    [new_candidates_list.append([candidates_list[cross_sample_list[idx * 2 + 1]][0].copy(), candidates_list[cross_sample_list[idx * 2]][1].copy()]) for idx in range(int(cross_sample_cnt / 2))]

    del roulette_df, candidates_list, cross_sample_list, copy_sample_list, j
    # +++ 돌연변이 +++
    new_candidates_list.sort()
    max_duplicate_cnt = 0
    mutate_cnt = int(math.ceil(candidates_len * 0.95)) if candidates_len * 0.05 > 1 else candidates_len - 1

    if (epoch != 1) & (epoch % 10 == 1): 
        duplicate_cnt = 0
        dupl_seq_list = []
        dupl_bun_list = []

        for candidate_list in new_candidates_list:
            if (candidate_list[0] == dupl_seq_list) & (candidate_list[1] == dupl_bun_list):
                duplicate_cnt += 1
            else:
                
                max_duplicate_cnt = duplicate_cnt if max_duplicate_cnt < duplicate_cnt else max_duplicate_cnt
                dupl_seq_list = candidate_list[0]
                dupl_bun_list = candidate_list[1]
                duplicate_cnt = 0

        # 50% 이상 하나로 수렴할 경우 절반을 돌연변이 시킴
        if max_duplicate_cnt / candidates_len >= 0.5:

            mutate_cnt = int(math.ceil(candidates_len * 0.5)) if candidates_len * 0.5 > 1 else candidates_len - 1

    # 돌연변이
    pick_mutation_list = random.sample(range(candidates_len), candidates_len - mutate_cnt)
    
    for idx in pick_mutation_list:
        
        mutate_candidate_list = set_mutate_candidate_list(new_candidates_list[idx])
        new_candidates_list[idx] = mutate_candidate_list
    
    return new_candidates_list

# 점수 계산 class
class setCandScore():
    def __init__(self,):
        self.standby_time = 2
        self.agg_distance = 0
        self.p2p_duration = 5
        self.p2d_duration = 3
        self.std_OE_duration = 45
        self.bundle_cnt = 0
        self.before_val = -1
        self.before_order_idx = -1
        
        self.group_id = 0
        self.seq_no = 0 
        
        self.fibonacci_weight = [1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155]
    
    def set_agg_distance(self, row):

        if row['orderNo'] == '-1':
            self.agg_distance = 0
        else : 
            self.agg_distance += row['deli_duration']

        agg_distance = self.agg_distance

        return agg_distance
    
    def _set_agg_distance(self, orderNo, deli_duration):

        if orderNo == '-1':
            self.agg_distance = 0
        else : 
            self.agg_distance += deli_duration

        agg_distance = self.agg_distance

        return agg_distance

    def set_bundle_cnt(self, seq_val):

        if (seq_val == -1) & (self.before_val >= 0):
            self.bundle_cnt += 1
        
        self.before_val = seq_val

    def set_seq_no(self, now_at, order_idx):
        seq_no = 0
        t_group_id = 0
        
        if (order_idx == -1):
            self.seq_no = 0
            if self.before_order_idx >= 0 :
                self.group_id += 1
            
            t_group_id = "-1"
            
        elif (self.before_order_idx == -1):
            self.seq_no = 0
            t_group_id = "{}_{:02d}".format(now_at.strftime('%y%m%d_%H%M'), int(self.group_id))
            
        else :
            self.seq_no += 1
            seq_no = self.seq_no
            t_group_id = "{}_{:02d}".format(now_at.strftime('%y%m%d_%H%M'), int(self.group_id))

        self.before_order_idx = order_idx

        return [t_group_id, seq_no]
    
    def set_candidate_score_df(self, now_at, candidate_list, real_df):
        df_at = dt.datetime.now()
        sub_candidate_list = candidate_list.copy()
        sub_candidate_list.append(-1)

        self.agg_distance = 0
        self.before_order_idx = -1
        self.group_id = 0
        self.seq_no = 0

        sub_real_df = real_df.loc[sub_candidate_list, :].copy()
        sub_real_df[['_x', '_y']] = sub_real_df[['x', 'y']].shift(1)
        sub_real_df.loc[:, 'group_id'] = "-1"
        sub_real_df.loc[:, 'seq_no'] = 0
        sub_real_df.fillna(0, inplace = True)
        
        sub_real_df.loc[:, ['group_id', 'seq_no']] = [self.set_seq_no(now_at, sub_candidate_list[order_idx]) for order_idx in range(len(sub_candidate_list))]
        
        sub_gr_real_df = sub_real_df.groupby(['group_id']).agg({'expect_cooking_end_at' : 'max'})
        sub_real_df.loc[:, 'expect_cooking_end_at'] = [pd.Timestamp(sub_gr_real_df.loc[_, 'expect_cooking_end_at'], tz = 'UTC') for _ in sub_real_df.group_id.values]
        
        sub_real_df['distance'] = m_dist_coord(sub_real_df['x'].values, sub_real_df['y'].values, sub_real_df['_x'].values, sub_real_df['_y'].values)
        sub_real_df['deli_duration'] = sub_real_df[['orderNo', 'distance']].apply(lambda x : (0 if x['orderNo'] == '-1' else x['distance'] * self.p2p_duration + self.p2d_duration), axis = 1)
        sub_real_df['deli_agg_duration'] = sub_real_df[['orderNo', 'deli_duration']].apply(lambda x : self.set_agg_distance(x), axis = 1)
        
        sub_real_df['order_to_standby_delta'] = sub_real_df['expect_cooking_end_at'].values - sub_real_df['orderDate'].values
        sub_real_df['order_to_standby_duration'] = sub_real_df['order_to_standby_delta'].values / np.timedelta64(1, 's') / 60
        
        sub_real_df['order_to_end_duration'] = sub_real_df['order_to_standby_duration'].values + sub_real_df['deli_agg_duration'].values
        
        del sub_gr_real_df
        
        return sub_real_df

    def set_candidates_score_list(self, now_at, candidate_list, real_df):
        
        sub_real_df = self.set_candidate_score_df(now_at, candidate_list, real_df)
        after_candidate_score_at = dt.datetime.now()
        sub_real_df['degree_weight'] = set_degree_weight(sub_real_df[['_x', '_y', 'x', 'y']].values)
        sub_real_df['delay_duration'] = [_ - self.std_OE_duration if _ > self.std_OE_duration else 0 for _ in sub_real_df['order_to_end_duration'].values]
        sub_real_df['_deli_duration'] = sub_real_df['seq_no'].apply(lambda x : self.fibonacci_weight[int(x)]) * sub_real_df['deli_duration'].values * sub_real_df['degree_weight'].values
        
        deli_duration = sub_real_df.deli_duration.sum() + self.bundle_cnt * self.standby_time
        _deli_duration = sub_real_df._deli_duration.sum() + self.bundle_cnt * self.standby_time
        delay_duration = sub_real_df.delay_duration.sum()
        
        return deli_duration, delay_duration, _deli_duration 

# 룰렛휠 판 만들기
class setRoulWheelDF():
    def __init__(self, ):
        
        self.start_score = 0
        self.end_score = 0
        self.idx = 0

    def set_roulette_df(self, score):
        
        self.end_score += score

        roulette_list = [score, self.start_score, self.end_score]
        self.start_score = self.end_score
        self.idx += 1

        return roulette_list

# 한번에 돌려야 하는 주문의 수 마다, 샘플의 갯수와 횟수를 다르게 함
# 10개의 주문 기준으로 한 경우라 주문의 갯수가 적으면 빨리 좋은 결과가 나올 수 있다고 생각되고, 더 많은 경우 더 많이 돌려야 좋은 결과가 나올 것으로 예상 됨
class setBundleDF():
    def __init__(self, now_at, real_df):
        self.now_at = now_at
        self.real_df = real_df
        self.order_cnt = len(real_df.index)
        self.max_epoch = 100 - (10 - self.order_cnt) * 10 if self.order_cnt < 15 else 150
        self.sample_cnt = 100
        self.max_duration = 45
        self.max_order_diff = 20
        
    def set_ga(self, candidates_list, dummy_df):

        print("{} 개 샘플로 {} 회 반복 실행".format(self.sample_cnt, self.max_epoch))
        
        epoch = 0
        now_at = self.now_at
        max_epoch = self.max_epoch
        
        while epoch < max_epoch:
            
            candidates_score_df = set_cand_score_df(now_at, epoch, candidates_list, dummy_df)
            
            new_candidates_list = set_new_candidate_list(epoch, candidates_list, candidates_score_df)
            
            candidates_list = new_candidates_list.copy()
            
            del new_candidates_list

            epoch += 1
            
        del epoch, now_at, max_epoch, candidates_score_df, dummy_df
            
        return candidates_list

    def set_best_candidate_list(self, run_at):
        
        now_at = self.now_at
        real_df = self.real_df
        sample_cnt = self.sample_cnt
        
        dummy_df = pd.DataFrame([[now_at, now_at, '-1', 0,0,0,0]] ,columns = ['orderDate', 'expect_cooking_end_at', 'orderNo', 'origin_lat', 'origin_lng', 'lat', 'lng'], index = [-1])
        dummy_df = real_df.append(dummy_df)

        # 극 좌표를 직교 좌표로 변경
        rectangular_coord_list = dummy_df.apply(lambda x : matrix_rotation(x['lat'], x['lng'], x['origin_lat'], x['origin_lng']), axis = 1)
        rectangular_coord_df = pd.DataFrame(list(rectangular_coord_list), columns = ['x', 'y'], index = dummy_df.index)
        
        dummy_df = dummy_df.merge(rectangular_coord_df, how = 'inner', left_index = True, right_index = True)
        
        # 하나만 있을 경우 유전자를 만들지 않고, 결과만 내놓음
        if len(real_df.index) > 1:
            candidates_list = set_candidates_list(sample_cnt, real_df)
            candidates_list = self.set_ga(candidates_list, dummy_df)
        else:
            candidates_list = [[[0], []]]
        
        t_candidate_len = len(candidates_list[0][0]) + math.floor(len(candidates_list[0][0]) / 2)
        
        candidates_list.sort()
        candidates_score_df = set_cand_score_df(now_at, 0, candidates_list, dummy_df)
        
        trans_candidates_list = [set_trans_candidate_list(candidate_list) for candidate_list in candidates_list]
        candidates_df = pd.DataFrame(trans_candidates_list, columns = [str(seq) for seq in range(0, t_candidate_len)])
        candidates_df = candidates_df.merge(candidates_score_df, how = 'inner', left_index = True, right_index = True)
        
        candidates_df = candidates_df.sort_values(['score', 'delay_score', 'deli_score'], ascending = [False, False, False])
        
        best_no = candidates_df.head(1).index[0]
        candidate_list = candidates_list[best_no]
        t_candidate_list = set_trans_candidate_list(candidate_list)

        cand_score = setCandScore()
        candidate_df = cand_score.set_candidate_score_df(now_at, t_candidate_list, dummy_df)
    
        confirm_candidate_df = candidate_df[candidate_df['group_id'].apply(lambda x : x in candidate_df[(candidate_df['order_to_end_duration'] >= self.max_duration) | (candidate_df['order_to_standby_duration'] >= self.max_order_diff)].group_id.values)][['orderNo', 'group_id', 'seq_no', 'distance']]
        
        candidate_df.loc[:, 'is_standby'] = 0
        candidate_df.loc[confirm_candidate_df.index, 'is_standby'] = 1
        
        candidate_df.loc[:, 'created_at'] = run_at
        _candidate_df = candidate_df.astype(str)
        
        del candidate_df, _candidate_df
        return confirm_candidate_df