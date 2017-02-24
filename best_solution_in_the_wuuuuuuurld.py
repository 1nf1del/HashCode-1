import numpy as np
from tqdm import tqdm

from time import gmtime, strftime

import fibonacci_heap_mod as fheap

from pqdict import minpq

from IO import *


def c2e_list(n_cache, endpoints):
    # create empty adjacency lists
    adjacency_list = []
    for i in range(n_cache):
        adjacency_list.append([])

    # fill adjacency lists
    for endpoint in endpoints:
        for cid, _ in endpoint.con:
            adjacency_list[cid].append(endpoint.id)

    return adjacency_list


def v2e_list(n_vid, n_end, requests):
    # create empty adjacency lists
    adjacency_list = []
    for i in range(n_vid):
        adjacency_list.append([])

    # fill adjacency lists
    for vid, eid, _ in requests:
        adjacency_list[vid].append(eid)

    return adjacency_list


def c2e_matrix(n_cache, endpoints):
    res = np.zeros(shape=(n_cache, len(endpoints)), dtype=np.bool)

    for endpoint in endpoints:
        for cid, _ in endpoint.con:
            res[cid][endpoint.id] = True

    return res


def v2e_matrix(n_vid, n_end, requests):
    res = np.zeros(shape=(n_vid, n_end), dtype=np.bool)

    for vid, eid, _ in requests:
        res[vid][eid] = True

    return res


def solution(n_vid, n_end, n_req, n_cache, s_cache, s_videos, endpoints, requests):
    # solutions
    cache = np.zeros(n_cache)
    videos_on_cache = [[] for _ in range(n_cache)]
    # compute scores
    scores = np.zeros(shape=(n_vid, n_cache), dtype=np.double)
    in_q = np.zeros(shape=(n_vid, n_cache), dtype=np.double)

    pq = fheap.Fibonacci_heap()
    entry_dict = {}

    v2e_m = v2e_matrix(n_vid, n_end, requests)
    v2e_l = v2e_list(n_vid, n_end, requests)
    c2e_l = c2e_list(n_cache, endpoints)
    c2e_m = c2e_matrix(n_cache, endpoints)

    for req in tqdm(requests):
        endpoint = endpoints[req.eid]
        for cid, lat in endpoint.con:
            d_latency = endpoint.lat - lat
            score = req.n * d_latency * (1.0 / s_videos[req.vid])
            scores[req.vid][cid] -= score

    for req in tqdm(requests):
        endpoint = endpoints[req.eid]
        for cid, _ in endpoint.con:
            if not in_q[req.vid][cid]:
                in_q[req.vid][cid] = True
                index = req.vid * n_cache + cid
                entry_dict[index] = pq.enqueue(index, scores[req.vid][cid])

    print((len(pq)) / (n_vid * n_cache), "in Queue.")

    # update from here on
    pbar = tqdm(total=len(pq))
    while len(pq) > 0:
        pbar.update(1)

        # get min entry
        entry = pq.dequeue_min()
        key, score = entry.get_value(), entry.get_priority()
        entry_dict.pop(key)

        # print and write intermediate solution score
        # if len(pq) % 5000 == 0:
        #     print(len(pq))
        #     print("Intermediate score: {0}".format(compute_solution_score(cache, videos_on_cache, requests,
        # endpoints)))
        #     write_solution(strftime("%Y%m%d-%H%M%S", gmtime()), cache, videos_on_cache)

        if score == float("-inf"):
            continue

        v, c = key // n_cache, key % n_cache
        if cache[c] + s_videos[v] <= s_cache:
            videos_on_cache[c].append(v)
            cache[c] += s_videos[v]

            # update scores for connected caches / videos
            for eid in range(n_end):
                # endpoint connects to cache
                if c2e_m[c][eid]:
                    # if video is requested by the endpoint
                    if v2e_m[v][eid]:
                        for cc in endpoints[eid].con:
                            scores[v][cc[0]] = float("-inf")

            # update the pq
            for k in entry_dict.keys():
                ss = entry_dict[k].get_priority()
                vv, cc = k // n_cache, k % n_cache
                new_score = scores[vv][cc]
                if ss != new_score:
                    # pq[k] = new_score
                    pq.decrease_key(entry_dict[k], new_score)
                    # print(entry_dict[k].get_priority())
                    # (_, vvvv, cccc) = heappop(pq)
                    # heappush(pq, (scores[vvvv][cccc], vvvv, cccc))

            # update scores for connected caches / videos
            for eid in set(c2e_l[c]) & set(v2e_l[v]):
                for cid, _ in endpoints[eid].con:
                    scores[v][cid] = float("-inf")
                    index = v * n_cache + cid
                    if index in entry_dict:
                        pq.decrease_key(entry_dict[index], scores[v][cid])


    pbar.close()

    return cache, videos_on_cache

def solution2(n_vid, n_end, n_req, n_cache, s_cache, s_videos, endpoints, requests):
    # solutions
    cache = np.zeros(n_cache)
    videos_on_cache = [[] for _ in range(n_cache)]
    # compute scores
    scores = np.zeros(shape=(n_vid, n_cache), dtype=np.double)
    in_q = np.zeros(shape=(n_vid, n_cache), dtype=np.double)

    pq = minpq()

    v2e_m = v2e_matrix(n_vid, n_end, requests)
    v2e_l = v2e_list(n_vid, n_end, requests)
    c2e_l = c2e_list(n_cache, endpoints)
    c2e_m = c2e_matrix(n_cache, endpoints)

    for req in tqdm(requests):
        ep = endpoints[req.eid]
        for c in ep.con:
            d_latency = ep.lat - c[1]
            score = req.n * d_latency * (1.0 / s_videos[req.vid])
            scores[req.vid][c[0]] -= score

    for req in tqdm(requests):
        ep = endpoints[req.eid]
        for c in ep.con:
            if not in_q[req.vid][c[0]]:
                in_q[req.vid][c[0]] = True
                # heappush(pq, (scores[req.vid][c[0]], req.vid, c[0]))
                index = req.vid * n_cache + c[0]
                pq[index] = scores[req.vid][c[0]]

    print((len(pq))/(n_vid * n_cache), "in Queue.")

    # update from here on
    while pq:
        # (s, v, c) = heappop(pq)
        key, s = pq.popitem()

        v, c = key // n_cache, key % n_cache
        if cache[c] + s_videos[v] <= s_cache:
            videos_on_cache[c].append(v)
            cache[c] += s_videos[v]

            print(len(pq))

            # update scores for connected caches / videos
            # for req in requests:
            #     if req.vid == v:
            #         ep = endpoints[req.eid]
            #         for ca in ep.con:
            #             d_latency = ep.lat - ca[1]
            #             score = req.n * d_latency * (1.0 / s_videos[req.vid])
            #             scores[req.vid][ca[0]] = 0  # score

            # for eid in set(c2e_l[c]) & set(v2e_l[v]):
            #     for cid, _ in endpoints[eid].con:
            #         scores[v][cid] = 0
            #         index = v * n_cache + cid
            #         if index in pq:
            #             pq[index] = scores[v][cid]

            # update scores for connected caches / videos
            print((v, c))
            print("before_")
            for eid in c2e_l[c]:
                # endpoint connects to cache
                # if c2e_m[c][eid]:
                    # if video is requested by the endpoint
                    if v2e_m[v][eid]:
                        for cid, _ in endpoints[eid].con:
                            idx = v * n_cache + cid
                            if scores[v][cid] != 0:
                                print((v, cid, scores[v][cid]), end=' -> ')
                                scores[v][cid] = 0
                                print((v, cid, scores[v][cid]))
                            # pq[v * n_cache + cc[0]] = scores[v][cc[0]]

            # update the pq
            print("after_")
            for (k, ss) in pq.items():
                vv, cc = k // n_cache, k % n_cache
                new_score = scores[vv][cc]
                if ss != new_score:
                    print((vv, cc))
                    pq[k] = new_score
                # (_, vvvv, cccc) = heappop(pq)
                # heappush(pq, (scores[vvvv][cccc], vvvv, cccc))

            sys.exit()

        if len(pq) % 100000 == 0:
            write_solution(strftime("%Y%m%d-%H%M%S", gmtime()), cache, videos_on_cache)

    return cache, videos_on_cache
