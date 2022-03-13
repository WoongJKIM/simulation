import simpy
import random
import datetime as dt
import math
import itertools

import numpy as np
import pandas as pd

from dependencies import make_bundle

order_df_columns = ['order_id', 'orderNo', 'orderDate', 'group_id', 'order_site', 'room_no', 'order_at', 'cooking_start_at', 'cooking_end_at', 'final_cooking_end_at', 'delivery_start_at', 'delivery_end_at', 'is_standby', 'is_delivery', 'lat', 'lng', 'origin_lat', 'origin_lng', 'rider_no', 'seq_no', 'distance']
rider_df_columns = ['start_at', 'end_at', 'rider_site', 'rider_no']

def set_distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c

    return distance

def set_manhattan_dist(x, y, origin_x, origin_y):
    
    return abs(origin_x - x) + abs(origin_y - y)

class Rider(object):
    def __init__(self, name, rider_site):
        self.name = name
        self.rider_site = rider_site
        self.is_return = 0
        self.is_delivery = 0
        self.origin_lat =  37.498 if rider_site == 'seocho' else 37.511036
        self.origin_lng = 127.0248 if rider_site == 'sinsa' else 127.020094
        
    def __repr__(self):
        
        return "{} rider_site {}".format(self.name, self.rider_site)

class simDelivery():
    
    def __init__(self, run_at, max_rider, run_time = 60):
        self.run_at = run_at
        self.max_rider = max_rider
        
        self.check_time = 0.1
        self.run_time = run_time
        self.restraunt_chef_no = 2
        self.make_bundle_period = 10
        self.rider_standby_time = 2.5
        self.cooking_time = 5
        self.expect_cooking_time = 15
        self.delivery_time_per_km = 7
        self.walking_time = 1
        self.order_df = pd.DataFrame(columns = order_df_columns)
        self.rider_df = pd.DataFrame(columns = rider_df_columns)
        
    def set_new_rider_df(self, env, riders):

        yield env.timeout(0)
        idx = 0
        for item in riders.items:
            rider_no = item.name
            rider_site = item.rider_site

            rider_dict = {'start_at' : env.now, 'rider_site' : rider_site, 'rider_no' : rider_no}
            self.rider_df.loc[idx, ['start_at', 'rider_site', 'rider_no']] = [env.now, rider_site, rider_no]
            idx += 1
        
    def set_process_bundle(self, env, start_at, run_time):
        make_cnt = 1
        while make_cnt * self.make_bundle_period < run_time + 60:
            yield env.timeout(self.make_bundle_period)
            print('{} - SET BUNDLE : {:2d}'.format((dt.datetime.now()).strftime("%H:%M:%S"), make_cnt))
            now_at = start_at + dt.timedelta(minutes = make_cnt * self.make_bundle_period)
            
            target_bundle_order_df = self.order_df[((self.order_df)['is_delivery'] == 0) & (self.order_df['is_standby'] == 0)][['orderDate', 'orderNo', 'origin_lat', 'origin_lng', 'lat', 'lng']].copy()
            target_bundle_order_df.reset_index(drop = True, inplace = True)
            target_bundle_order_df.loc[:, 'expect_cooking_end_at'] = target_bundle_order_df['orderDate'].apply(lambda x : x + dt.timedelta(minutes = self.expect_cooking_time) if x + dt.timedelta(minutes = self.expect_cooking_time) > now_at else now_at)
            
            if len(target_bundle_order_df) > 0:
                
                set_bundle_df = make_bundle.setBundleDF(now_at, target_bundle_order_df)
                bundle_order_df = set_bundle_df.set_best_candidate_list(self.run_at)
                
                for idx, row in bundle_order_df.iterrows():
                    self.order_df.loc[self.order_df[self.order_df['orderNo'] == row['orderNo']].index, ['group_id', 'seq_no', 'is_standby', 'distance']] = [row['group_id'], row['seq_no'], 1, row['distance']]
            
            print('{} - END BUNDLE '.format((dt.datetime.now()).strftime("%H:%M:%S")))
            make_cnt += 1
        
    def set_process_order(self, env, order_id, orderNo, orderDate, order_at, order_site, room_no, origin_lat, origin_lng, lat, lng, room_res, riders):
        is_stop = True
        
        yield env.timeout(order_at)
        self.order_df.loc[int(order_id), ['order_id', 'orderNo', 'orderDate', 'group_id', 'order_site', 'room_no', 'order_at', 'cooking_start_at', 'cooking_end_at', 'final_cooking_end_at', 'delivery_start_at', 'delivery_end_at', 'is_standby', 'is_delivery', 'lat', 'lng', 'origin_lat', 'origin_lng']] = [order_id, orderNo, orderDate, '-1', order_site, room_no, order_at, -1.0, 1440.0, 1440.0, -1.0, -1.0, 0, 0, lat, lng, origin_lat, origin_lng]
        print('ORDER : {:.2f} - order_id {} / order_site {}'.format(env.now, order_id, order_site))
        
        with room_res.request() as req:
            yield req
            self.order_df.loc[int(order_id), 'cooking_start_at'] = env.now
            print('COOKING : {:.2f} - order_id {} / order_site {}'.format(env.now, order_id, order_site))
            
            yield env.timeout(self.cooking_time)
            cooking_end_at = env.now
            self.order_df.loc[int(order_id), 'cooking_end_at'] = cooking_end_at
            print('COOKED : {:.2f} - order_id {} / order_site {}'.format(env.now, order_id, order_site))
        
        while is_stop:
            yield env.timeout(self.check_time)
            
            is_standby = self.order_df.loc[int(order_id), 'is_standby']
            is_final_cooked = 0
            
            if is_standby == 1:
                group_id = self.order_df.loc[order_id, 'group_id']
                
                order_id_list = self.order_df[self.order_df['group_id'] == group_id].order_id.values
                max_final_cooked_at = self.order_df[self.order_df['group_id'] == group_id].cooking_end_at.max()
                is_final_cooked = 1 if cooking_end_at == max_final_cooked_at else 0
                
                is_stop = False
                
            self.order_df.loc[order_id, 'is_final_cooked'] = is_final_cooked
        
        if is_final_cooked == 1:
            
            print("STAND BY : {:.2f} - gruop_id : {}(order_cnt {}) / order_site {}".format(env.now, group_id, len(order_id_list), order_site))
            print("---"*10)
            self.order_df.loc[order_id_list, 'final_cooking_end_at'] = cooking_end_at
            gr_order_df = self.order_df.loc[order_id_list].copy()
            gr_order_df.sort_values(['seq_no'], ascending = True, inplace = True) 
            print(gr_order_df.head(20))
            while True:
                try:
                    #대기 하는 라이더 중 지점으로 복귀 했고, 배달이 없고, 묶음 배송 지점과 같은 지점에서 대기하는 라이더 들에 자원을 점유함
                    rider = yield riders.get(lambda x : (x.is_return == 0) & (x.is_delivery == 0) & (x.rider_site == order_site))
                    
                    print("START DELIVER : {:.2f} - group_id : {}(order_num {}) / order_site {} / rider_name {}".format(env.now, group_id, order_id, order_site, rider.name))
                    
                    self.order_df.loc[order_id_list, 'delivery_start_at'] = env.now
                    self.order_df.loc[order_id_list, 'rider_no'] = rider.name
                    self.order_df.loc[order_id_list, 'is_delivery'] = 1
                    
                    break
                    
                except Exception as ex: # 에러 종류
                    pass
            
            yield env.timeout(self.rider_standby_time)
            for idx, row in gr_order_df.iterrows():
                
                delivery_time = row['distance'] * self.delivery_time_per_km + self.walking_time
                yield env.timeout(delivery_time)
                print("DELIVERY END : {} - group_id : {}({})".format(env.now, group_id, idx))
                order_idx = self.order_df[self.order_df['orderNo'] == row['orderNo']].index[0]
                self.order_df.loc[order_idx, 'delivery_end_at'] = env.now
                
                if row['seq_no'] == gr_order_df.seq_no.max():
                    rider.is_delivery = 0
                    rider.is_return = 1
                    riders.put(rider)
            
                    delivery_time = set_distance([row['origin_lat'], row['origin_lng']], [row['lat'], row['lng']]) * self.delivery_time_per_km
                    
                    # rider_site = "gk-kangnam"
            
            yield env.timeout(delivery_time)
            rider.is_return = 0
            rider_no = rider.name
            
            rider_index = list(self.rider_df[self.rider_df['rider_no'] == rider_no].index)
            max_idx = max(list(self.rider_df.index))
            rider_max_idx = max(rider_index)
            print("RETURN RIDER : {} - group_id : {} / rider_no : {}".format(env.now, group_id, rider_no))
            print("---" * 10)
            self.rider_df.loc[rider_max_idx, 'end_at'] = env.now
            self.rider_df.loc[max_idx + 1, ['rider_no', 'rider_site', 'start_at']] = [rider_no, rider.rider_site, env.now]
            
    def set_order(self, start_at, real_df):
        env = simpy.Environment()
        
        riders = simpy.FilterStore(env, capacity = self.max_rider)
        riders.items = [Rider("rider_{:02d}".format(i), "seocho" if random.random() <= 1 else "sinsa") for i in range(self.max_rider)]
        
        room_res_dict = {}
        for room_no in list(set(real_df.room_no.values)):
            room_res_dict[room_no] = simpy.Resource(env, capacity = self.restraunt_chef_no)
        
        env.process(self.set_process_bundle(env, start_at, self.run_time))
        env.process(self.set_new_rider_df(env, riders))
        
        for idx, row in real_df[real_df['order_at'] < self.run_time].iterrows():

            order_id = int(idx)
            orderNo = row['orderNo']
            orderDate = row['orderDate']
            order_at = row['order_at']
            order_site = row['site']
            room_no = row['room_no']
            
            origin_lat = row['origin_lat']
            origin_lng = row['origin_lng']
            lat = row['lat']
            lng = row['lng']

            order_process = self.set_process_order(env, order_id, orderNo, orderDate, order_at, order_site, room_no, origin_lat, origin_lng, lat, lng, room_res_dict[room_no], riders)
            env.process(order_process)

        env.run(until = self.run_time + 60)
        
        self.rider_df.fillna(self.run_time + 60, inplace = True)
        
        self.rider_df.loc[:, 'created_at'] = self.run_at
        self.order_df.loc[:, 'created_at'] = self.run_at
       
        # self.order_df.to_csv("order_df_{}.csv".format((self.run_at).strftime('%Y%m%d%H%M')))
        # self.rider_df.to_csv("rider_df_{}.csv".format((self.run_at).strftime('%Y%m%d%H%M')))
                        
#         spreadsheet_key = "spreadsheet_key"
#         order_title = "{}_주문내역_{}".format(start_at.strftime('%Y%m%d'), (dt.datetime.now()).strftime('%m%d%H%M'))
#         rider_title = "{}_라이더내역_{}".format(start_at.strftime('%Y%m%d'), (dt.datetime.now()).strftime('%m%d%H%M'))
        
#         sh = send_spreadsheet.sendSpreadsheet()
#         sh.set_send_spreadsheet(spreadsheet_key, order_title, 1, order_df)
#         sh.set_send_spreadsheet(spreadsheet_key, title_title, 1, rider_df)