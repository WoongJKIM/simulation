import numpy as np
import pandas as pd
import random

import datetime as dt

import os
import imutils
import re

import cv2 
import matplotlib.pyplot as plt
# from imgRead import imgRead
import pytesseract
# import easyocr
from PIL import ImageFont, ImageDraw, Image

# 텍스트 매칭률을 판단하기 위해서
from difflib import SequenceMatcher
from jamo import h2j, j2hcj

default_insight_type_dict = {
    "게시물 인사이트" : {
        "개요" : {
            "도달한 계정" : {'axis' : 'row'}, 
            "참여한 계정" : {'axis' : 'row'}, 
            "프로필 활동" : {'axis' : 'row'},
            },
        "도달" : {
            "노출" : {'axis' : 'row'},
            "홈" : {'axis' : 'row'},
            "탐색 탭" : {'axis' : 'row'},
            "기타" : {'axis' : 'row'},
            "도달한 계정" : {'axis' : 'upper_col'},
            "팔로워" : {'axis' : 'upper_col'},
            "팔로워가 아닌 사람" : {'axis' : 'upper_col'},
        },
        "참여" : {
            "노출" : {'axis' : 'row'},
            "홈" : {'axis' : 'row'},
            "탐색 탭" : {'axis' : 'row'},
            "기타" : {'axis' : 'row'},
            "참여한 계정" : {'axis' : 'upper_col'},
            "팔로워" : {'axis' : 'upper_col'},
            "팔로워가 아닌 사람" : {'axis' : 'upper_col'},
        },
    },
    "릴스 인사이트" : {
        "릴스 인사이트" : {
            "Instagram 및 Facebook 재생 횟수" : {'axis' : 'row'},
            "Instagram 및 Facebook 좋아요" : {'axis' : 'row'},
        },
        "Instagram" : {
            "도달한 계정" : {'axis' : 'upper_col'}, 
            "재생 횟수" : {'axis' : 'row'}, 
            "좋아요" : {'axis' : 'row'}, 
            "저장" : {'axis' : 'row'}, 
            "댓글" : {'axis' : 'row'}, 
            "공유" : {'axis' : 'row'}
        },
        "Facebook" : {
            "도달한 계정" : {'axis' : 'upper_col'}, 
            "재생 횟수" : {'axis' : 'row'}, 
            "좋아요" : {'axis' : 'row'}, 
            "저장" : {'axis' : 'row'}, 
            "댓글" : {'axis' : 'row'}, 
            "공유" : {'axis' : 'row'}
        },
        "도달" : {
            "도달한 사람 수" : {'axis' : 'upper_col'}, 
            "재생 횟수" : {'axis' : 'row'}, 
            "좋아요" : {'axis' : 'row'}, 
            "저장" : {'axis' : 'row'}, 
            "댓글" : {'axis' : 'row'}, 
            "공유" : {'axis' : 'row'}
        },
        "콘텐츠 활동" : {
            "콘텐츠 활동" : {'axis' : 'row'}, 
            "재생 횟수" : {'axis' : 'row'}, 
            "좋아요" : {'axis' : 'row'}, 
            "저장" : {'axis' : 'row'}, 
            "댓글" : {'axis' : 'row'}, 
            "공유" : {'axis' : 'row'}
        },
    }
}

insight_target_list = ['게시물 인사이트', '릴스 인사이트', 'Reel insight', 'Post insight', '인사이트', 'Insights']
kor_key_list = ['게시물 인사이트', '개요', '도달한 계정', '참여한 계정', '프로필 활동', '도달', '도달한 계정', '팔로워', '팔로워가 아닌 사람', '노출', '홈', '탐색 탭', '해시태그', '기타', '참여', '참여한 계정', '팔로워', '팔로워가 아닌 사람',
                '릴스 인사이트', 'Instagram 및 Facebook 재생 횟수', 'Instagram 및 Facebook 좋아요', '재생 횟수', '좋아요', '저장', '댓글', '공유', '콘텐츠 활동', '프로필', '게시물 반응', '팔로워가 아닌 참여한 사람', '참여한 팔로워', 
                '콘텐츠 상호 작용', '릴스 상호 작용', '게시물 반응', '프로필 활동', '프로필 방문', '이메일 보내기 버튼 누름', '광고', '이 게시물 홍보하기', '도달한 사람 수']
eng_key_list = ['Reel insight', 'Reach', 'Instagram', 'Facebook', 'People Reached', 'Content Interactions', 'Likes', 'Saves', 'Shares', 'Comments', 'Post insight', 'Overview', 'Accounts reached', 
                'Accounts engaged', 'Profile activity', 'Followers', 'Non-Followers', 'Impressions', 'From Home', 'From Profile', 'From Explore', 'From Other', 'Engagement', 'Insights']

def set_imshow(img, img_title):
    show_ratio = 0.8
    h, w = img.shape[:2]
    cv2.imshow(img_title, cv2.resize(img, (int(w * show_ratio), int(h * show_ratio))))

def set_img_binary(img):
    h, w = img.shape
    # 사진이 기본 모드 일 경우 반전, 아닐 경우 유지
    if img[0][0] < 100:
        print(1)
        binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]
    else:
        print(2)
        binary = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY_INV)[1]
        
    set_imshow(binary, "binary_image")

    return binary

def set_img_distance_transform(img):
     
    # 흑백 이미지의 distransform 값을 추출
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    # set_imshow(dist, "Dist - 1")

    # normalize the distance transform such that the distances lie in
    # the range [0, 1] and then convert the distance transform back to
    # an unsigned 8-bit integer in the range [0, 255]
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (dist * 255).astype("uint8")
    
    # 결과를 이진화 함
    binary = cv2.threshold(dist, 2, 255, cv2.THRESH_BINARY)[1]
    set_imshow(binary, "Dist - 2")

    return binary

def set_img_morphology_ex(img):
    # 침식 연산을 사용해 모폴로지 방법으로 글자 주변에 노이즈를 제거함
    # apply an "opening" morphological operation to disconnect components
    # in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 1))
    morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    set_imshow(morph, "Morphology")

    return morph

def set_mask(img, x_thread, y_thread):

    box_df = set_bounding_box_df(img, x_thread, y_thread)

    margin_thread = 5
    h, w = img.shape
    set_margin_bounding_box(box_df, w, h, margin_thread)

    # allocate memory for the convex hull mask, draw the convex hull on
    # the image, and then enlarge it via a dilation
    mask = np.zeros(img.shape, dtype="uint8")

    for idx, rect in box_df.iterrows():
        mask = cv2.rectangle(mask,(rect['min_x'], rect['min_y']),(rect['max_x'], rect['max_y']), 255, -1)     

    set_imshow(mask, "Mask")

    final = cv2.bitwise_and(img, img, mask=mask)

    # 최종 값을 이진화한다.
    final = cv2.threshold(final, 3, 255, cv2.THRESH_BINARY)[1]
    set_imshow(final, "Final")

    return final

def set_fit_bounding_box(x):
    is_fit = False
    if ((x['max_x'] - x['min_x']) + (x['max_y'] - x['min_y']) >= 20) & ((x['max_y'] - x['min_y']) > 15) :
         is_fit = True

    return is_fit

def set_margin_bounding_box(df, w, h, margin_thread):
     
    # 바운딩 박스에 마진을 줌
    df.loc[:, 'min_x'] = df.min_x.apply(lambda x : int(0) if x - margin_thread < 0 else int(x - margin_thread))
    df.loc[:, 'min_y'] = df.min_y.apply(lambda x : int(0) if x - margin_thread < 0 else int(x - margin_thread))
    df.loc[:, 'max_x'] = df.max_x.apply(lambda x : int(w) if x + margin_thread > w else int(x + margin_thread))
    df.loc[:, 'max_y'] = df.max_y.apply(lambda x : int(h) if x + margin_thread > h else int(x + margin_thread))

    return df

# 이미지에 사물을 검출해 바운딩 박스를 만들고 x와 y좌표에서 임계값 안에 있는 바운딩 박스를 합치는 작업하는 코드
def set_bounding_box_df(img, x_thread, y_thread):
    # 이미지의 width와 height를 가져옴
    h, w = img.shape

    # 이미지 내에 물체를 찾아 좌표를 가져옴(x, y, width, height)
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 최초 설정된 바운딩 박스 값을 저장하는 리스트
    rects_df_lists = []

    # 바운딩 박스의 idx를 설정
    cnt_list_idx = []
    cnt_idx = 0

    # 바운딩 박스의 값을 리스트에 저장
    for cnt in cnts:
        (x, y, width, height) = cv2.boundingRect(cnt)
        # & (y > int(h*0.03)) & (y + height < int(h * 0.91))
        if (width + height >= 20) :
            rects_df_lists.append((x, y, x + width, y + height))
            cnt_list_idx.append(cnt_idx)
            cnt_idx += 1

    # 최초 바운딩 박스를 데이터 프레임에 저장
    sample_box_df = pd.DataFrame(rects_df_lists, columns = ['min_x', 'min_y', 'max_x', 'max_y'], index = cnt_list_idx)
    box_lists = []

    # 박스의 값들의 인덱스 정보를 들고 있음
    sample_box_df_idx_list = sample_box_df.index

    #이미지에서 x와 y 범위 내에서 바운딩 박스들을 합치기 위한 설정 값
    xThr = x_thread
    yThr = y_thread	

    #최초 바운딩 박스 중 하나를 임의로 선택해 그 박스 인근에 있는 박스를 통합
    #남아있는 최초 바운딩 박스가 없을 경우까지 박스를 통합
    while len(sample_box_df_idx_list) > 0:
        
        idx = random.choice(sample_box_df_idx_list)
        i = [idx]

        curr_min_x = sample_box_df.loc[idx, 'min_x']
        curr_max_x = sample_box_df.loc[idx, 'max_x']
        curr_min_y = sample_box_df.loc[idx, 'min_y']
        curr_max_y = sample_box_df.loc[idx, 'max_y']

        while len(i) > 0:
            cand_min_x_lists = sample_box_df.min_x.values
            cand_max_x_lists = sample_box_df.max_x.values
            cand_min_y_lists = sample_box_df.min_y.values
            cand_max_y_lists = sample_box_df.max_y.values

            i, j = np.where((xThr >= (cand_min_x_lists[:, None] - curr_max_x)) & (xThr >= (curr_min_x - cand_max_x_lists[:, None])) & (yThr >= (cand_min_y_lists[:, None] - curr_max_y)) & (yThr >= (curr_min_y - cand_max_y_lists[:, None])))
            
            if len(i) < 1:
                box_lists.append((curr_min_x, curr_min_y, curr_max_x, curr_max_y))
                break

            pop_i_df = sample_box_df.iloc[i, :]
            pop_lists = pop_i_df.index
        
            sub_df = sample_box_df.loc[pop_lists, :]
            curr_min_x = min(curr_min_x, sub_df.min_x.min())
            curr_min_y = min(curr_min_y, sub_df.min_y.min())
            curr_max_x = max(curr_max_x, sub_df.max_x.max())
            curr_max_y = max(curr_max_y, sub_df.max_y.max())
        
            pop_idx_lists = sub_df.index

            trans_t_df = sample_box_df.T
            [trans_t_df.pop(x) for x in pop_idx_lists]
            sample_box_df = trans_t_df.T

            sample_box_df_idx_list = sample_box_df.index

            del sub_df, trans_t_df, pop_i_df, pop_lists, cand_min_x_lists, cand_max_x_lists, cand_min_y_lists, cand_max_y_lists
        
    # 통합된 바운딩 박스를 저장
    sub_box_df = pd.DataFrame(box_lists, columns = ['min_x', 'min_y', 'max_x', 'max_y'])

    # 통합된 바운딩 박스 중 작은 것은 제외함
    box_df = sub_box_df[(sub_box_df.apply(lambda x : set_fit_bounding_box(x), axis = 1))]
    # box_df = sub_box_df
    del sub_box_df
    
    # 바운딩 박스들을 정렬함
    box_df.sort_values(['max_y', 'min_x'], ascending = [True, True], inplace = True)

    return box_df

def chk_vaild_text_box_df(box_df):
    spacial_char_pattern = re.compile(r'[^\w\s]')
    
    for idx, row in box_df.iterrows():
        res_text = spacial_char_pattern.sub('', row['res_text'])

        box_df.loc[idx, 'res_text'] = res_text
        if len(res_text) > 1:
            box_df.loc[idx, 'is_vaild_text'] = 1

    return box_df

def set_match_text(res_text, key_list, acc_thread):
    acc_ratio = 0.0
    matching_res = ""
    
    for key in key_list:
        sub_acc_ratio = SequenceMatcher(None, j2hcj(h2j(key)), j2hcj(h2j(res_text))).ratio()
        
        if  (sub_acc_ratio >= acc_thread) & (acc_ratio < sub_acc_ratio) & (abs(len(res_text) - len(key)) < 5):
            matching_res  = key
            acc_ratio = sub_acc_ratio
            
    # print("글자 비교 단계 - 원문 : ", res_text , ", 리스트 : ", key_list, ", 결과 : ", matching_res, ", 정확도 : ", acc_ratio)

    return matching_res

def set_match_text_v2(res_text, key_list, is_kor, acc_thread):
    acc_ratio = 0.0
    matching_res = ""
    
    # 컬럼과 동일한 값일 경우 비교하는 값을 체크 하지 않음
    if res_text in key_list:
        matching_res = res_text

    elif len(res_text) < len('Instagram 및 Facebook 재생 횟수'):
        for key in key_list:
            if is_kor == 1:
                #한국어의 경우 자모를 분리하는 로직을 태운 후 두 단어가 비슷한지 확인함
                sub_acc_ratio = SequenceMatcher(None, j2hcj(h2j(key)), j2hcj(h2j(res_text))).ratio()
            else:
                sub_acc_ratio = SequenceMatcher(None, key, res_text).ratio()
            
            if  (sub_acc_ratio >= acc_thread) & (acc_ratio < sub_acc_ratio) & (abs(len(res_text) - len(key)) < 5):
                matching_res  = key
                acc_ratio = sub_acc_ratio
                
    # print("글자 비교 단계 - 원문 : ", res_text , ", 리스트 : ", key_list, ", 결과 : ", matching_res, ", 정확도 : ", acc_ratio)

    return matching_res

def set_ocr_res_dict_v2(box_df, w, h):
    res_dict = {}
    box_idx_lists = box_df[box_df['is_col'] == 1].index
    res_col_lists = []

    for box_idx in box_idx_lists:
        row_no = box_df.loc[box_idx, 'row_no']
        min_x = box_df.loc[box_idx, 'min_x']
        max_x = box_df.loc[box_idx, 'max_x']
        res_text = box_df.loc[box_idx, 'res_text']
        
        
        if min_x < w * 0.15:
            val_idx_list = box_df[(box_df['row_no'] == row_no) & ([x != box_idx for x in box_df.index])].index
            
        else :
            val_idx_list = box_df[(box_df['row_no'] == row_no - 1) & (20 > (min_x - box_df['max_x'])) & (20 > (box_df['min_x'] - max_x))].index
        
        res_text = res_text + '_{}'.format(row_no) if res_text in res_col_lists else res_text

        res_dict[res_text] = box_df.loc[val_idx_list[0], 'res_text'] if len(val_idx_list) > 0 else 0
        res_col_lists.append(res_text)

        box_df.loc[box_idx, 'is_used'] = 1
        if len(val_idx_list) > 0 :
            box_df.loc[val_idx_list[0], 'is_used'] = 1


    box_gr_df = box_df.groupby(['row_no']).agg(col_cnt=("res_text", "count"))

    multi_col_idx_lists = box_gr_df[box_gr_df['col_cnt'] > 3].index

    for col_idx in multi_col_idx_lists:
        if box_gr_df.loc[col_idx, 'col_cnt'] == 4:
            col_list = ['좋아요_i', '댓글_i', 'DM_i', '저장_i']
            col_idx_list = box_df[box_df['row_no'] == col_idx].index
            
            col_idx = 0
            for idx in col_idx_list:
                
                res_dict[col_list[col_idx]] = box_df.loc[idx, "res_text"]
                box_df.loc[idx, "is_used"] = 1
                col_idx += 1
        else:
            col_list = ['재생_i', '좋아요_i', '댓글_i', 'DM_i', '저장_i']
            col_idx_list = box_df[box_df['row_no'] == col_idx].index
            
            col_idx = 0
            for idx in col_idx_list:
                
                res_dict[col_list[col_idx]] = box_df.loc[idx, "res_text"]
                box_df.loc[idx, "is_used"] = 1
                col_idx += 1

    etc_res_text_list = []
    for idx, res in box_df[box_df['is_used'] == 0].iterrows():
        etc_res_text_list.append(res['res_text'])

    res_dict['etc'] = etc_res_text_list

    return res_dict


def set_ocr_res_dict(box_df):
    res_dict = {}
    sub_res_dict = {}

    insight_text = '인사이트'
    insight_type_pattern = re.compile(insight_text)

    insight_type = ''
    insight_1st_key_list = default_insight_type_dict.keys()
    insight_2nd_key_list = []
    insight_3rd_key_list = []
    insight_2nd_key = ''
    insight_3rd_key = ''

    for idx, rect in box_df.iterrows():
        # set_imshow(sub_bounding_box, "{}-BoundingBox".format(idx))
        res_text = rect['res_text']

        # box_df.loc[idx, 'is_string'] = 1 if re.search('[가-힣a-zA-Z]+', res_text) else 0
        # if re.search('', res_text)

        if (insight_type == '') & ((rect['is_kor'] == 1) | (rect['is_eng'] == 1)) & (rect['is_number'] == 0):
            acc_thread = 0.8
            insight_type = set_match_text(res_text, insight_1st_key_list, acc_thread)
            print('res_text : ', res_text, ', 인사이트 타입 : ', insight_type)

            if insight_type != '':
                insight_2nd_key_list  = default_insight_type_dict[insight_type].keys()

                res_dict['dict_type'] = insight_type
            
        elif (insight_type != '') & ((rect['is_kor'] == 1) | (rect['is_eng'] == 1)):
            acc_thread = 0.6
            sub_insight_2nd_key = set_match_text(res_text, insight_2nd_key_list, acc_thread)
            insight_2nd_key = insight_2nd_key if sub_insight_2nd_key == '' else sub_insight_2nd_key

            if (insight_2nd_key == sub_insight_2nd_key) & (sub_insight_2nd_key != ''):
                sub_res_dict ={}

            if ((insight_2nd_key != '') & (sub_insight_2nd_key == '')) | (sub_insight_2nd_key == '콘텐츠 활동'):
                
                acc_thread = 0.6
                insight_3rd_key_list = default_insight_type_dict[insight_type][insight_2nd_key].keys()
                insight_3rd_key = set_match_text(res_text, insight_3rd_key_list, acc_thread)
                print("res_text : ", res_text, "유형 : ", insight_type, ", 대제목 : ", insight_2nd_key, ", 소제목 : ", insight_3rd_key)

                if insight_3rd_key == '':
                    pass

                elif (insight_3rd_key != '') & (default_insight_type_dict[insight_type][insight_2nd_key][insight_3rd_key]['axis'] == 'row') :
                    val_idx = box_df[(box_df['row_no'] == rect['row_no']) & ([x != idx for x in box_df.index])].index[0]
                    sub_res_dict[insight_3rd_key] = box_df.loc[val_idx, 'res_text']
                    # res_dict.update({insight_2nd_key : {insight_3rd_key : box_df.loc[val_idx, 'res_text']}})

                    # i, j = np.where((xThr >= (cand_min_x_lists[:, None] - curr_max_x)) & (xThr >= (curr_min_x - cand_max_x_lists[:, None])) & (yThr >= (cand_min_y_lists[:, None] - curr_max_y)) & (yThr >= (curr_min_y - cand_max_y_lists[:, None])))
                elif (insight_3rd_key != '') & (default_insight_type_dict[insight_type][insight_2nd_key][insight_3rd_key]['axis'] == 'upper_col') :
                    val_idx = box_df[(box_df['row_no'] == rect['row_no'] - 1) & ([x != idx for x in box_df.index]) & (20 > (rect['min_x'] - box_df['max_x'])) & (20 > (box_df['min_x'] - rect['max_x']))].index[0]
                    sub_res_dict[insight_3rd_key] = box_df.loc[val_idx, 'res_text']
                    # res_dict.update({insight_2nd_key : {insight_3rd_key : box_df.loc[val_idx, 'res_text']}})
        
            if (insight_type != '') & (insight_2nd_key != '') & (insight_3rd_key != ''):
                print("sub_res_dict : ", sub_res_dict)
                res_dict[insight_2nd_key] = sub_res_dict

            insight_3rd_key = ''

    return res_dict

def set_box_row_no_df(box_df):

    box_df.loc[:, 'row_no'] = -1

    box_idx_lists = box_df.index
    box_idx = box_idx_lists[0]
    row_thread = 10
    row_no = 0

    while box_idx in box_idx_lists:

        adjacent_idx_lists = box_df[(box_df['avg_y'].apply(lambda x : abs(x - box_df.loc[box_idx, 'avg_y']) < row_thread)) & (box_df['row_no'] == -1)].index

        if len(adjacent_idx_lists) > 1:
            box_df.loc[adjacent_idx_lists, 'row_no'] = row_no
            max_adjacent_idx = max(adjacent_idx_lists)
            box_idx = max_adjacent_idx
        else:
            box_df.loc[adjacent_idx_lists, 'row_no'] = row_no
            row_no += 1
            box_idx += 1
            
    box_df = box_df.reset_index()
    box_df = box_df.drop('index', axis = 1)

    box_df.sort_values(['row_no', 'min_x'], ascending = [True, True])

    return box_df

def set_text_to_img(img, x_thread, y_thread, oem_idx, psm_idx):
    cfg_option = f'--oem {oem_idx} --psm {psm_idx} -c preserve_interword_spaces=1'
    print("-------------------------------------------------")
    print("이미지 영역별 출력")
    # print(cfg_option)
    h, w = img.shape

    acc_thread = 0.66

    box_df = set_bounding_box_df(img, x_thread, y_thread)
    margin_thread = 10
    box_df = set_margin_bounding_box(box_df, w, h, margin_thread)
    box_df.loc[:, 'is_valid_text'] = 0

    img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)[1]

    special_char_pattern = re.compile(r'[^가-힣0-9a-zA-Z\-\s\:]')

    for idx, rect in box_df.iterrows():
        sub_bounding_box = img[rect[1] : rect[3], rect[0] : rect[2]]
        # set_imshow(sub_bounding_box, "{}-BoundingBox".format(idx))

        res_text = pytesseract.image_to_string(sub_bounding_box, lang='kor+eng', config = cfg_option)
        res_text = special_char_pattern.sub('', res_text)
        res_text = res_text.strip()

        is_col = 0
        is_kor = 0
        is_eng = 0
        is_number = 0

        if len(res_text) > 1 :
            is_valid_text = 1

            is_kor = 1 if re.search('[가-힣]+', res_text) else 0
            is_eng = 1 if re.search('[a-zA-Z]+', res_text) else 0
            is_number = 1 if re.search('[0-9]+', res_text) else 0

            box_df.loc[idx, 'is_valid_text'] = is_valid_text
            box_df.loc[idx, 'is_kor'] = is_kor
            box_df.loc[idx, 'is_eng'] = is_eng
            box_df.loc[idx, 'is_number'] = is_number

        elif len(res_text) > 0 :
            
            is_valid_text = 1 if re.search('[가-힣0-9]+', res_text) else 0

            is_kor = 1 if re.search('[가-힣]+', res_text) else 0
            is_eng = 0
            is_number = 1 if re.search('[0-9]+', res_text) else 0

            box_df.loc[idx, 'is_valid_text'] = is_valid_text
            box_df.loc[idx, 'is_kor'] = is_kor
            box_df.loc[idx, 'is_eng'] = is_eng
            box_df.loc[idx, 'is_number'] = is_number
        
        if (is_kor == 1) | (is_eng == 1):
            key_list = kor_key_list if is_kor == 1 else eng_key_list
            sub_res_text = set_match_text_v2(res_text, key_list, is_kor, acc_thread)

            if sub_res_text != "":

                res_text = sub_res_text
                is_col = 1
        
        box_df.loc[idx, 'is_col'] = is_col
        box_df.loc[idx, 'res_text'] = res_text
        box_df.loc[idx, 'avg_y'] = (rect['max_y'] + rect['min_y']) / 2

    box_df = box_df[box_df['is_valid_text'] == 1]

    box_df = box_df.sort_values(['avg_y', 'min_x'], ascending = [True, True])
    
    box_df = box_df.reset_index()
    box_df = box_df.drop('index', axis = 1)
    
    box_df = set_box_row_no_df(box_df)
    box_df = box_df.sort_values(['row_no', 'min_x'], ascending = [True, True])

    box_df = box_df.reset_index()
    box_df = box_df.drop('index', axis = 1)

    return box_df

def set_img_to_result_img(img_path):

    ocr_to_img_start_dt = dt.datetime.now()
    print(os.path.join('./sample', img_path))
    default_img_path = os.path.join('./sample', img_path)
    
    # 이미지 파일 읽음
    img = cv2.imread(default_img_path)
    set_imshow(img, "raw_image")

    # 컬러 이미지의 노이즈 제거
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # set_imshow(img, "denoise_image")

    # 컬러 이미지 흑백으로 전환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 사진 원본의 사이즈 저장
    h, w= gray.shape

    # binary = set_img_binary(gray)
    is_dark = 0
    # 사진이 기본 모드 일 경우 반전, 아닐 경우 유지
    if gray[0][0] < 100:
        is_dark = 1
        print(1)
        binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    else:
        print(2)
        binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)[1]
    
    # set_imshow(binary, "binary_image")

    # 사진의 영상값 히스토그램 분포 확인
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # plt.title("histogram")
    # plt.xlabel("Bin")
    # plt.ylabel("Frequency")
    # plt.plot(np.arange(256), hist, color='b')
    # plt.show()

    # dist = set_img_distance_transform(binary)

    morph = set_img_morphology_ex(binary)

    # 바운딩 박스의 합치기를 위한 임계값 설정
    x_thread = 30
    y_thread = 3

    final = set_mask(morph, x_thread, y_thread)

    oem_idxs = [1] #range(1, 4)
    psm_idxs = [7] #range(0, 13) / [7, 8, 11]

    for oem_idx in oem_idxs:
        for psm_idx in psm_idxs:
            
            # try :
            box_df = set_text_to_img(final, x_thread, y_thread, oem_idx, psm_idx)
            box_df.loc[:, 'is_used'] = 0

            # print(box_df.head(50))

            ocr_res_dict = set_ocr_res_dict_v2(box_df, w, h)
            # ocr_res_dict = set_ocr_res_dict(box_df)

            print(ocr_res_dict)

            res_img = Image.fromarray(img)
            # font = ImageFont.truetype('/Users/ji-woongkim/fonts/gulim.ttc', 40)
            draw = ImageDraw.Draw(res_img)

            for idx, rect in box_df.iterrows():

                draw.rectangle(((rect[0], rect[1]), (rect[2], rect[3])), outline = (255,0,0), width = 2)
                font = ImageFont.truetype('/Users/ji-woongkim/fonts/gulim.ttc', 40)
                draw.text((int(rect[0]) + 5, rect[3] - 10), str(rect['row_no']) + "-" + str(rect['res_text']), font = font, fill = (255,0,0), )
                
            ocr_to_img_end_dt = dt.datetime.now()
            diff_sec = (ocr_to_img_end_dt - ocr_to_img_start_dt).total_seconds()

            ocr_res_dict['실행 시간'] = '{:.4} 초'.format(diff_sec)

            font = ImageFont.truetype('/Users/ji-woongkim/fonts/gulim.ttc', 30)
            ocr_str = str(ocr_res_dict)
            ocr_str_length = len(ocr_str) 
            str_length_hist = 60
            str_length = 0
            ocr_idx = 0

            text_color = (255, 255, 255) if is_dark == 1 else (0,0,0)

            while str_length < ocr_str_length:
                str_length = str_length_hist * ocr_idx
                
                draw.text((0, (30 * (ocr_idx))), ocr_str[str_length : str_length + str_length_hist], font = font, fill = text_color, )
                
                ocr_idx += 1
            
            print("실행 시간 : ", diff_sec, "초")
            # cv2.imshow('final_result_bouding_box', res_img)
            plt.figure(f'oem_idx : {oem_idx}, psm_idx : {psm_idx}'  ,figsize = (10,15))
            plt.title(f'{img_path} - oem_idx : {oem_idx}, psm_idx : {psm_idx}')
            plt.imshow(res_img)
            print(f'./res/{img_path}-oem_idx_{oem_idx}_psm_idx_{psm_idx}.jpg')
            plt.savefig(f'./res/{img_path}-oem_idx_{oem_idx}_psm_idx_{psm_idx}.jpg')
            # except:
            #     print(f'실패 - oem_idx : {oem_idx}, psm_idx : {psm_idx}')

    
    plt.show()
    
    # 키보드 입력을 기다린 후 모든 영상창 닫기
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

img_paths = ['1681907550593.png', '1681907550722.png', '1681907550787.png', '1681907550840.png', '1681907555229.png', '1681907560530.png', 'IMG_2351.jpg', 'IMG_2352.png', 'IMG_2353.png', 'IMG_2354.png',
             '립제이 인사이트1.png', '립제이 인사이트2.png', '박민주 인사이트1.png', '박민주 인사이트2.png', '박민주 인사이트3.png', '인사이트1.png', '인사이트2png.png', 'IMG_3291.jpeg', 'KakaoTalk_Photo_2023-05-22-17-05-07 001.jpeg',
             'KakaoTalk_Photo_2023-05-31-15-27-26 001.png', 'KakaoTalk_Photo_2023-05-31-15-27-26 002.png', 'KakaoTalk_Photo_2023-05-31-15-27-39.png', 'KakaoTalk_Photo_2023-05-31-15-27-48.png',
             'KakaoTalk_Photo_2023-05-31-15-27-56.png', 'KakaoTalk_Photo_2023-05-31-15-28-07 002.jpeg', 'KakaoTalk_Photo_2023-05-31-15-28-21.jpeg', 'KakaoTalk_Photo_2023-05-31-15-28-38 001.jpeg',
             'KakaoTalk_Photo_2023-05-31-15-28-38 002.jpeg', 'KakaoTalk_Photo_2023-05-31-15-28-54.png', 'KakaoTalk_Photo_2023-05-31-15-29-15.png', 'KakaoTalk_Photo_2023-06-05-13-19-17 001.jpeg',
             'KakaoTalk_Photo_2023-06-05-13-19-18 002.jpeg', 'KakaoTalk_Photo_2023-06-05-13-26-36 001.png', 'KakaoTalk_Photo_2023-06-05-13-26-36 002.png', 'KakaoTalk_Photo_2023-06-05-13-27-08 001.jpeg',
             'KakaoTalk_Photo_2023-06-05-15-29-43.jpeg', 'KakaoTalk_Photo_2023-06-05-15-29-47.jpeg',  'KakaoTalk_Photo_2023-06-05-15-30-18 003.png']
## 잘 안된 것
# img_paths = ['KakaoTalk_Photo_2023-05-31-15-28-21.jpeg']
# img_paths = ['KakaoTalk_Photo_2023-06-05-13-19-17 001.jpeg']
img_paths = ['IMG_2354.png']
for img_path in img_paths:
    set_img_to_result_img(img_path)
    

    