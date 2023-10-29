#!/usr/bin/env python
"""Module to generate a higher-order network (HON) in a supervised learning setting.

This script is an adaptation of GrowHON [https://github.com/sjkrieg/growhon]. It's
a bit of a frankenstein but gets the job done.

Overview: this program takes as a text file containing trajectories as input.
It returns a graph---specifically, a higher-order network) as output.
The output is formatted as a weighted edgelist and can be read using other packages
like networkx. Currently, it only supports a binary classification setting, 
where '0.' and '1.' are the text versions of the negative and positive labels, respectively.

Please see the README.txt on the github [[https://github.com/sjkrieg/growhonsup]
for more details on the input formats.
"""

# =================================================================
# MODULES REQUIRED FOR CORE FUNCTIONALITY
# =================================================================
import argparse
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
from collections import defaultdict, deque, Counter
from itertools import islice, permutations, chain, combinations
from math import log, log2, ceil
from multiprocessing import cpu_count
from tqdm import tqdm
# =================================================================
# MODULES USED FOR LOGGING AND DEBUGGING
# =================================================================
from os import getpid
from psutil import Process
from time import perf_counter, strftime, gmtime
# import logging last or other modules may adjust the logging level
import logging
# =================================================================

__version__ = '0.1'
__author__ = 'Steven Krieg'
__email__ = 'skrieg@nd.edu'

class HONTree():
    def __init__(self, 
                 k, 
                 inf_name=None, 
                 num_cpus=0,
                 z=1.0,
                 minsup=10,
                 inf_delimiter=' ', 
                 inf_num_prefixes=1, 
                 order_delimiter='|',
                 log_name=None, 
                 seq_grow_log_step=1000, 
                 par_grow_log_interval=0.2,
                 dotf_name=None,
                 noskip=False):
        
        # basic initializations and parameter setting
        logging.basicConfig(level=logging.INFO, filename=log_name,
                            format='%(levelname)-8s  %(asctime)23s  ' + 
                            '%(runtime)10s  %(rss_size)6dMB  %(message)s')
        self.logger = logging.LoggerAdapter(logging.getLogger(), self._LogMap())
        self.root = self._HONNode(tuple())
        self.active_nodes = 0
        self.k = k
        self.z = z
        self.minsup = minsup
        self.nmap = {}
        self.nmap[tuple()] = self.root
        self.inf_delimiter = inf_delimiter
        self.inf_num_prefixes = inf_num_prefixes
        self.seq_grow_log_step = seq_grow_log_step
        self._HONNode.order_delimiter = order_delimiter
        self.noskip = noskip

        self.num_cpus = num_cpus if num_cpus else cpu_count()
        if inf_name: 
            self.grow(inf_name)
    
    
    def close_log(self):
        logging.shutdown()
    
    
    def grow(self, inf_name, num_cpus=None, inf_delimiter=None, inf_num_prefixes=None):
        """The main method used to grow the tree.
        """
        
        # reassign object variables if they were passed explicitly
        if num_cpus: self.num_cpus = num_cpus
        if inf_delimiter: self.inf_delimiter = inf_delimiter
        if inf_num_prefixes: self.inf_num_prefixes = inf_num_prefixes
        
        self.logger.info('Growing tree with max order {}...'.format(self.k))
        self._grow_sequential(inf_name)
        self.logger.info('Growing successfully completed!')
    
    
    def extract(self, otf_name=None, delimiter=' '):
        """Converts the HONTree into a weighted edgelist via DFS.
        """
        
        self.logger.info('Extracting edgelist...')
        edgelist = {}
        # __extract_helper modifies edgelist in place
        log_step = ceil(0.1 * len(self.root.children))
        for i, c in enumerate(self.root.children.values()):
            if not i % log_step:
                self.logger.info('Extracting {}/{} branches...'.format(i, len(self.root.children)))
            self.__extract_helper(c, edgelist)
        self.logger.info('Extracted {:,d} nodes and {:,d} edges. Formatting...'.format(len(set(chain.from_iterable(edgelist.keys()))), len(edgelist)))
        
        # edgelist is structured as (u, v): w, where
        # u is the source node, v is the destination node
        # and w is the edge weight
        edgelist = [edge[0] + delimiter + edge[1] + delimiter 
                  + str(weight) for edge, weight in edgelist.items()]

        if otf_name:
            try:
                with open(otf_name, 'w+') as otf: otf.write('\n'.join(edgelist))
                self.logger.info('Edgelist successfully written to {}.'.format(otf_name))
            except:
                self.logger.info('Exception: Could not open {}.'.format(otf_name))
        else:
            #return edgelist 
            self.logger.info('Skipping edgelist write.')
        
    # if the child does not already exist, add it
    # returns a reference to the child node
    def _add_child(self, parent, child_label, deg=1):
        """If the child node does not already exist, add it. Returns a reference to the child node.
        """
        if child_label in parent.children:
            child = parent.children[child_label]
        else:
            child = self._HONNode(child_label, parent)
            parent.children[child_label] = child
            if child.order <= self.k:
                self.nmap[child.name] = child
            self.active_nodes += 1
        parent.out_deg += deg
        child.in_deg += deg
        return child

    def _grow_sequential(self, inf_name):
        """Grow the tree. This function oversees the following steps:
        1) parsing the input file
        2) count the occurrence of all possible higher-order combinations for each order (k)
        3) exclude all the combinations that don't 
            a) have a sufficiently high information gain
            AND
            b) meet the minimum frequency threshold
        4) use all the preserved combinations to grow a tree
        """
        
        self.logger.info('Reading input from {}...'.format(inf_name))
        with open(inf_name,'r') as inf:
            lines = [line.strip().split(self.inf_delimiter)[self.inf_num_prefixes:] for line in inf]
        
        labels = [line[-1] for line in lines]
        trajs = [line[:-1] for line in lines]
        self.logger.info('Loaded {:,d} trajectories!'.format(len(lines)))
        
        # compute all the first-order combos and information gains
        combos_k1 = self._get_combos(trajs, 1, labels=labels)
        self.gains = [self._compute_gains(labels, combos_k1)[0]]
        min_freq_set_prev = None
        # keep track of the nodes to keep at each order, having None first will be important later
        nodes_to_keep = [None, set(self.gains[0].keys())]
        
        # iterate over the higher-orders
        for k in range(2, self.k+1):
            # get all combos of length k with label suffixes 
            combos_curr = self._get_combos(trajs, k, labels=labels, min_freq_set_prev=min_freq_set_prev)
            # compute information gain for all combos
            gains_curr, min_freq_set_prev = self._compute_gains(labels, combos_curr)
            # keep a node if it passes the significance test
            nodes_to_keep_curr = set(gains_curr.keys())
            nodes_to_keep.append(nodes_to_keep_curr)
            # garbage collection - this helps prevent too much memory bloat
            self.logger.info('Collecting garbage...')
            del combos_curr
            gc.collect()
            self.logger.info('Garbage collection done!')
            self.gains.append(gains_curr)
            
        self.logger.info('found {} significant nodes at each level'.format([len(n) for n in nodes_to_keep[1:]]))
        # ensure we are preserving graph connectivity for the higher-order nodes we want to add
        for k in range(len(nodes_to_keep)-1, 2, -1):
            cur_parents = {u[:-1] for u in nodes_to_keep[k]}
            nodes_to_keep[k-1].update(cur_parents)
        self.logger.info('keeping {} nodes at each level (to preserve graph connectivity)'.format([len(n) for n in nodes_to_keep[1:]]))
        
        self.nodes_to_keep = nodes_to_keep
        # iterate over each level of nodes and add them to the tree
        # use None first to add all first-order nodes as base nodes (children of root)
        for k, V_k in enumerate(nodes_to_keep):
            # use lines here to also include the label as a destination node
            # the reason we recompute the combos here is to help preserve memory
            # the first time we compute combos, it is purely to compute information gain for each node
            # now we are retrieving the list of out-edges, which means the combos are order k+1
            # at higher orders this can get really inefficient
            # but by including the nodes_to_keep argument, we skip a lot of the unnecessary computations
            edges = self._get_combos(lines, k+1, nodes_to_keep=V_k)
            
            for u, w in tqdm(edges.items(), desc='Adding out-edges to the graph'):
                self._add_child(self.nmap[u[:-1]], u, deg=w)

            self.logger.info('Added {:,d} out-edges from k={} nodes (tree contains {:,d} total)!'.format(len(edges), k, len(self)))


    def _get_combos(self, lines, k, labels=None, nodes_to_keep=None, min_freq_set_prev=None):
        """Generate all the k-th order combos of nodes in lines. This is non-parallel implementation.
        """
        
        # if we don't allow for the skip step, we don't need to parallelize - get those combos
        if self.noskip:
            return self._get_combos_noskip(lines, k, labels=labels, nodes_to_keep=nodes_to_keep, min_freq_set_prev=min_freq_set_prev)
    
        # we don't always need to run in parallel
        if self.num_cpus > 1 and (k > 2 or nodes_to_keep):
            return self._get_combos_parallel(lines, k, labels=labels, nodes_to_keep=nodes_to_keep, min_freq_set_prev=min_freq_set_prev)
        
        if nodes_to_keep is not None and len(nodes_to_keep) == 0:
            return Counter()
        
        heartbeat_interval=10
        steplen = len(lines)//heartbeat_interval
        steps = [steplen*i for i in range(heartbeat_interval)] + [len(lines)]
        combos = Counter()
        
        # there are a few scenarios to consider:
        # first, if labels is not None, we need to generate all the combos that lead to each outcome
        # we use this condition when we are computing information gains
        if labels is not None:
            for i in tqdm(range(len(steps)-1), desc='Generating label combos for k={}...'.format(k)):
                if min_freq_set_prev:
                    # wow this list comprehension is a doozy, sorry
                    # basically, we are trying to generate all the combinations of elements in each line + label destinations using the itertools.combinations generator
                    # we use dict.fromkeys() to drop duplicates, since it's faster than set
                    # we ALSO use .issubset() to check if each combination meets the minimum frequency parameter
                    # finally, we use steps[] to divide the computation into chunks for logging purposes
                    # the computationally fastest way to do this is to call a single Counter() over a chain.from_iterable object containing ALL the lines
                    # however, this can take a long time without providing any feedback to the user
                    # so wrapping it in the tqdm loop gives a signal to the user without calling .update() on Counter after every single line, which would be terribly slow
                    cur_combos = Counter(chain.from_iterable([[tuple(c)+(labels[j],) for c in combinations(dict.fromkeys(lines[j]).keys(), k) if set(combinations(c,k-1)).issubset(min_freq_set_prev)] for j in range(steps[i],steps[i+1])]))
                else:
                    cur_combos = Counter(chain.from_iterable([[tuple(c)+(labels[j],) for c in combinations(dict.fromkeys(lines[j]).keys(), k)] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
        # if labels is None but nodes_to_keep is not, we are computing the list of out-edges from each node
        elif nodes_to_keep is not None:
            for i in tqdm(range(len(steps)-1), desc='Generating out-edges for k={}...'.format(k-1)):
                # this list comprehension follows similar logic to the one above
                cur_combos = Counter(chain.from_iterable([[tuple(c) for c in combinations(dict.fromkeys(lines[j]).keys(), k) if tuple(c)[:-1] in nodes_to_keep] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
        # if labels is None, we are computing the list of out-edges from each node
        # if nodes_to_keep is also None, we keep everything and don't need to check whether certain nodes should be preserved or not
        else:
            for i in tqdm(range(len(steps)-1), desc='Generating all combos for k={}...'.format(k)):
                cur_combos = Counter(chain.from_iterable([[tuple(c) for c in combinations(dict.fromkeys(lines[j]).keys(), k)] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
            
        return combos
    
    
    def _get_combos_parallel(self, lines, k, labels=None, nodes_to_keep=None, min_freq_set_prev=None):
        """Generate all the k-th order combos of nodes in lines. This is a parallel implementation.
        """
        inq, outq = mp.Queue(), mp.Queue()
        # done_count tracks workers that have terminated; success_count counts the number of input lines that have been processed
        done_count, success_count = 0, 0
        # keep track of how long it has been since we heard from a worker
        lastmsg = perf_counter()
        self.logger.info('Growing combos for k={} using {} worker processes...'.format(k, self.num_cpus))
        
        if labels is None:
            labels = [None for _ in lines]
        
        # use inq to pass all the lines and labels to workers
        for line, label in zip(lines, labels):
            inq.put((line, label))
        # add a flag at the end of the q to let the workers know they can terminate
        for i in range(self.num_cpus):
            inq.put((None, None))

        combos = []
        workers = [mp.Process(target=self._get_combos_worker, args=(inq, outq, k, nodes_to_keep, min_freq_set_prev)) for n in range(self.num_cpus)]
        [w.start() for w in workers]
        
        # loop indefinitely until we know all the workers have finished
        while done_count < self.num_cpus:
            try:
                msg = outq.get(timeout=10)
                if msg is not None: # 1 means we finished a single trajectory
                    combos.append(msg)
                    success_count += 1
                else: # none means a worker has terminated
                    done_count += 1

                if perf_counter() - lastmsg > 5:
                    self.logger.info('{:,d} / {:,d} lines processed... ({} workers active)'.format(success_count, len(lines), self.num_cpus-done_count))
                    lastmsg = perf_counter()
            
            except KeyboardInterrupt:
                [w.terminate() for w in workers]
                raise SystemExit
            # if we didn't receive a response from a worker process, print a quick update to confirm the main process is still alive
            except:
                self.logger.info('{:,d} / {:,d} lines processed... ({} workers active)'.format(success_count, len(lines), self.num_cpus-done_count))
        self.logger.info('{:,d} / {:,d} lines processed -- done!'.format(success_count, len(lines)))
        [w.join() for w in workers]
        combo_counts = sum([len(c) for c in combos])
        self.logger.info('Generating counts from {:,d} combos... this might take a while...'.format(combo_counts))
        
        # if the combos are too long, we can divide them into chunks to provide better
        # feedback to users about how long the counting will take
        if combo_counts > 10**8:
            nsteps = combo_counts//10**8
            steplen = len(combos)//nsteps
            steps = [steplen*i for i in range(nsteps)] + [len(combos)]
            final_counts = Counter()
            for i in tqdm(range(len(steps)-1), desc='Generating combos...'):
                final_counts.update(Counter(chain.from_iterable(combos[steps[i]:steps[i+1]])))
        # if the combos are small enough, just do them all in one shot
        else:
            final_counts = Counter(chain.from_iterable(combos))
        self.logger.info('Done! Cleaning up garbage...')
        del combos
        gc.collect()
        self.logger.info('Done! Returning and preparing to compute gains...')
        return final_counts
        
    def _get_combos_noskip(self, lines, k, labels=None, nodes_to_keep=None, min_freq_set_prev=None):
        """This function is for computing a version of the graph without skip-steps.
        """
        def slices(line):
            return [line[i:i+k]for i in range(len(line)-k+1)]
            
        heartbeat_interval=10
        steplen = len(lines)//heartbeat_interval
        steps = [steplen*i for i in range(heartbeat_interval)] + [len(lines)]
        combos = Counter()
        
        if labels is not None:
            for i in tqdm(range(len(steps)-1), desc='Generating label combos for k={}...'.format(k)):
                cur_combos = Counter(chain.from_iterable([[tuple(c)+(labels[j],) for c in slices(list(dict.fromkeys(lines[j])))] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
        elif nodes_to_keep is not None:
            for i in tqdm(range(len(steps)-1), desc='Generating out-edges for k={}...'.format(k-1)):
                cur_combos = Counter(chain.from_iterable([[tuple(c) for c in slices(list(dict.fromkeys(lines[j]))) if tuple(c)[:-1] in nodes_to_keep] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
        else:
            for i in tqdm(range(len(steps)-1), desc='Generating all combos for k={}...'.format(k)):
                cur_combos = Counter(chain.from_iterable([[tuple(c) for c in slices(list(dict.fromkeys(lines[j])))] for j in range(steps[i],steps[i+1])]))
                combos.update(cur_combos)
            
        return combos
    
    
    def _get_combos_worker(self, inq, outq, k, nodes_to_keep, min_freq_set_prev):
        """Worker function for _get_combos_parallel
        """
        
        line, label = inq.get()
        # use these variables to help control memory usage
        i_def = 10**8
        i = i_def
        
        while line is not None:
            # if labels are provided, we want to compute the information gain of each combo and label destination
            if label is not None:
                if min_freq_set_prev:
                    cur_combos = [tuple(c)+(label,) for c in combinations(dict.fromkeys(line).keys(), k) if self._check_freq(c, k, min_freq_set_prev)]
                else:
                    cur_combos = [tuple(c)+(label,) for c in combinations(dict.fromkeys(line).keys(), k)]
            # if labels are not provided then we are computing out-edges for the graph
            elif nodes_to_keep is not None:
                cur_combos = [tuple(c) for c in combinations(dict.fromkeys(line).keys(), k) if tuple(c)[:-1] in nodes_to_keep]
            else:
                cur_combos = [tuple(c) for c in combinations(dict.fromkeys(line).keys(), k)]
            
            outq.put(cur_combos)
            i -= len(cur_combos)
            # clean up memory every so often
            if i < 0:
                i = i_def
                gc.collect()
            line, label = inq.get()
        outq.put(None)
    
    
    def _check_freq(self, c, k, min_freq_set_prev):
        if c[1:] not in min_freq_set_prev or c[:-1] not in min_freq_set_prev:
            return False
        for i in range(1, k-1):
            if c[:i] + c[i+1:] not in min_freq_set_prev:
                return False
        return True
        
        
    def _compute_gains(self, labels, combos):
        """Supervising function for computing information gains for each higher-order combo.
        Ensure keys in combos are tuples whose last element is '0.' or '1.'
        For a given node u and labels Y, we compute IG(Y, u) as h(Y) - h(Y|u)
        Where h(Y) is the entropy of Y and h(Y|u) is the conditional entropy of Y given u.
        """
        
        # this is a multiprocessing implementation, even if we only use a single worker
        min_freq_set = set()
        c0 = sum([label=='0.' for label in labels])
        c1 = len(labels)-c0
    
        gains = {}
        min_freq_set = set()
        keys = set(key[:-1] for key in combos.keys())
        inq, outq = mp.Queue(), mp.Queue()
        lastmsg = perf_counter()
        
        self.logger.info('Initializing workers...')
        # send all the combos and counts to workers
        for uprime in sorted(keys, key=len):
            cu0, cu1 = combos.get(uprime+('0.',), 0), combos.get(uprime+('1.',), 0)
            if cu0+cu1 >= self.minsup or len(uprime) == 1:
                inq.put((uprime, cu0, cu1))
                min_freq_set.add(uprime)
        for i in range(self.num_cpus):
            inq.put((None, None, None))
         
        workers = [mp.Process(target=self._compute_gain_worker, args=(c0, c1, inq, outq)) for n in range(self.num_cpus)]
        [w.start() for w in workers]
        success_count, done_count = 0, 0
        
        self.logger.info('Computing gains for {:,d} nodes using {} worker processes...'.format(len(min_freq_set), self.num_cpus))
        # loop until all the workers have terminated
        while done_count < self.num_cpus:
            try:
                msg = outq.get(timeout=10)
                if msg is None:
                    done_count += 1
                else: 
                    success_count += 1
                    if msg: # 2 len tuple comes as (uprime, zscore) pair; 1 len tuple means we don't save
                        uprime, z = msg
                        keep = True
                        # check that the gain of the new dependency is greater than its ancestors, 
                        # e.g. if a|b|c doesn't have higher gain than b|c, it's not worth keeping
                        for k in range(len(uprime)-2, -1):
                            if z < self.gains[k].get(uprime[-(k+1):], 0):
                                keep = True
                                break
                        if keep:
                            gains[msg[0]] = msg[1]
                    
                if perf_counter() - lastmsg > 5:
                    self.logger.info('{:,d} / {:,d} gains computed... ({} workers active)'.format(success_count, len(min_freq_set), self.num_cpus-done_count))
                    lastmsg = perf_counter()
            
            except KeyboardInterrupt:
                [w.terminate() for w in workers]
                raise SystemExit
            # if we didn't receive a response from a worker process, print a quick update to confirm the main process is still alive
            except:
                self.logger.info('{:,d} / {:,d} gains computed... ({} workers active)'.format(success_count, len(min_freq_set), self.num_cpus-done_count))
        self.logger.info('{:,d} / {:,d} gains computed -- done!'.format(success_count, len(min_freq_set)))  
        [w.join() for w in workers] 
        return gains, min_freq_set
    
    
    def _compute_gain_worker(self, c0, c1, inq, outq):
        """Worker function for computing information gain.
        For a given node u and labels Y, we compute IG(Y, u) as h(Y) - h(Y|u)
        Where h(Y) is the entropy of Y and h(Y|u) is the conditional entropy of Y given u.
        Here we also compute the z-statisitc 
        """
        def hxy(pxs, pxys):    
            hu = 0
            for pxy, px in zip(pxys, pxs):
                if pxy: hu -= pxy*log2(pxy/px)
            return hu
            
        m = c0 + c1
        p0 = c0 / m
        p1 = 1 - p0
        # base entropy
        hy = -(p0*log2(p0) + p1*log2(p1))
        rng = np.random.default_rng(777)
        
        uprime, cu0, cu1 = inq.get()
        while uprime is not None:
            # compute h(Y|u)
            cu = cu0 + cu1
            cnu = m - cu
            pxs = [c/(cu+cnu) for c in [cu, cu, cnu, cnu]]
            igxy = hy - hxy(pxs, [c/m for c in [cu0, cu1, c0-cu0, c1-cu1]])
            
            noise_sample = rng.choice([0,1], size=(100,cu), replace=True, p=[p0, p1])
            cu1_sample = noise_sample.sum(axis=1)
            cu0_sample = cu - cu1_sample
            ig_noise = np.array([hy - hxy(pxs, [c/m for c in [cu0_curr, cu1_curr, c0-cu0_curr, c1-cu1_curr]]) for cu0_curr, cu1_curr in zip(cu0_sample, cu1_sample)])
            if ig_noise.std() > 0:
                tstat = (igxy - ig_noise.mean()) / ig_noise.std()
            else:
                tstat = 0
            if tstat >= self.z or len(uprime) == 1:
                outq.put((uprime, igxy))
            else:
                # 1 length tuple indicates to manager that we don't want to save
                outq.put(False)
            uprime, cu0, cu1 = inq.get()
        outq.put(None)
    # =================================================================
    # ALL CODE BELOW THIS THROUGH THE END OF THE HONTREE DEFINITION
    # IS TAKEN DIRECTLY FROM THE ORIGINAL IMPLEMENTATION OF GROWHON
    # =================================================================
    # =================================================================
    # BEGIN SUPPORTING (_ and __) FUNCTIONS
    # =================================================================
    def __len__(self):
        return self.active_nodes
    
    def __str__(self):
        """Calls a recursive helper to traverse and print the tree.
        """
        return self.__str_helper(self.root)
    
    def __str_helper(self, node):
        """A recursive DFS helper for printing the tree.
        """
        if node:
            s = node.dump() if node != self.root else ''
            for child in node.children.values():
                s += self.__str_helper(child)
            return s
            
        
    def __extract_helper(self, node, edgelist):
        """Used by the extract() method to find the edge destination.
        
        Description:
            Each edge must be checked to ensure the integrity of flow
            in the HON is preserved. If an edge destination would
            result in a "dead end," the edge is redirected to the
            original destination's lower-order counterpart.
  
        Parameters: 
            node: the node whose destination we want to verify
            
        Returns:
            The label to be included as the edge destination
        """
        # if node.in_deg==0, it was pruned
        # if node.order==0, it is not a sequence destination
        if node.in_deg > 0:
            for child in node.children.values():
                self.__extract_helper(child, edgelist)
            if node.order > 0:
                #while (node.in_deg <= 0 or node.orphan) and node.order > 1:
                #    if self.verbose: self.logger.info('Trying to adopt {}'.format(str(node)))
                #    node = self._get_lord_match(node)
                v = node
                while v.out_deg == 0 and v.order > 0:
                    v = self._get_lord_match(v)
                edgelist[(node.parent.get_label_full(), v.get_label_full())] = node.in_deg
            

    def _get_lord_match(self, hord_node):
        """Used to find a node's lower-order counterpart.
        
        Description:
            This method is used by several others to find a 
            higher-order node's lower-order counterpart in O(1) time.
            This is done by truncating the first element (oldest
            history) from the higher-order node's label, then finding
            the new label in the nmap.

        Parameters: 
            hord_node (_HONNode): the higher-order node
            
        Returns:
            (_HONNode) the object reference to the lower-order node
        """
        if not hord_node.order: return None
        i = 1
        while hord_node.name[i:] not in self.nmap:
            i += 1
        return self.nmap[hord_node.name[i:]]


    # =================================================================
    # END SUPPORTING FUNCTIONS
    # =================================================================
    # =================================================================
    # BEGIN _HONNode DEFINITION
    # =================================================================
    class _HONNode:
        """The definition of all objects inserted into HONTree.
        
        Description:
            Each _HONNode represents a sequence from the input data and
            an edge in the output HON.

        Static Variables:
            order_delimiter (char): for labelling higher order nodes
        """
        # initialize to -1 to discount root node
        order_delimiter = None
        
        def __init__(self, name='', parent=None):
            self.name = name
            self.order = len(self.name) - 1
            self.in_deg = 0
            self.out_deg = 0
            self.parent = parent
            self.children = {}
            self.marked = False # used during pruning
            self.checked_for_merge = False
            
        def __str__(self):
            return ('{}[{}:{}]'.format(self.get_label_full(), self.in_deg, self.out_deg))

        def dump(self):
            return '------->' * (len(self.name) - 1) + str(self) + '\n'
        
        def get_label_full(self):
            return HONTree._HONNode.order_delimiter.join(reversed([str(c) for c in self.name]))
        
        def get_label_short(self):
            return self.name[-1]

    # =========================================================================
    # END _HONNode DEFINITION
    # =========================================================================
    # =========================================================================
    # BEGIN _LogMap DEFINITION
    # =========================================================================
    class _LogMap():
        """Internal class used to standardize logging.
        """

        def __init__(self):
            self.start_time = perf_counter()
            self._info = {}
            self._info['runtime'] = self.get_time_seconds
            self._info['rss_size'] = self.get_rss
        
        def __getitem__(self, key):
            return self._info[key]()
        
        def __iter__(self):
            return iter(self._info)
        
        def get_time(self):
            return strftime('%H:%M:%S', gmtime(perf_counter() - self.start_time))
        
        def get_time_seconds(self):
            return '{:.2f}s'.format(perf_counter() - self.start_time)
        
        def get_rss(self):
            return(Process(getpid()).memory_info().rss >> 20)
            
    # =========================================================================
    # END _LogMap DEFINITION
    # =========================================================================
# =================================================================
# END HONTree DEFINITION
# =================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infname', help='source path + file name')
    parser.add_argument('otfname', help='destination path + file name, use ! to avoid save')
    parser.add_argument('k', help='max order to use in growing the HON', type=int)
    parser.add_argument('-w', '--numcpus', help='number of workers', type=int, default=16)
    parser.add_argument('-x', '--minsup', help='minimum frequency for a pattern to be kept', type=int, default=100)
    parser.add_argument('-z', '--z', help='significance threshold for IG', type=float, default=10.0)
    parser.add_argument('-p', '--infnumprefixes', help='number of prefixes for input sequences', 
                        type=int, default=1)
    parser.add_argument('-di', '--infdelimiter', help='delimiter for entities in input sequences', 
                        default=' ')
    parser.add_argument('-do', '--otfdelimiter', help='delimiter for output network', default=' ')
    parser.add_argument('-o', '--logname', help='location to write log output', 
                        default=None)
    parser.add_argument('-lsg', '--logisgrow',
                        help='logging interval for sequential growth', type=int, default=10000)
    parser.add_argument('--noskip', help='no skip connections', action='store_true', default=False)
    parser.add_argument('--saveigs', help='location to save information gains as .csv file', default=None)
    args = parser.parse_args()

    t1 = HONTree(args.k, 
                 args.infname,
                 z=args.z,
                 minsup=args.minsup,
                 inf_num_prefixes=args.infnumprefixes, 
                 inf_delimiter=args.infdelimiter,
                 num_cpus=args.numcpus, 
                 log_name=args.logname,
                 seq_grow_log_step=args.logisgrow,
                 noskip=args.noskip,
                 )
    t1.extract(args.otfname if args.otfname != '!' else None, args.otfdelimiter)
    if args.saveigs:
        print('Saving IGs to {}...'.format(args.saveigs))
        pd.concat([pd.DataFrame.from_dict(g, orient='index', columns=['ig']) for g in t1.gains]).to_csv(args.saveigs)
    t1.close_log()
