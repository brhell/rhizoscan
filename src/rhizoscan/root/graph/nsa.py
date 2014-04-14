"""
Package that contains the definition of the NodeList, SegmentList and AxeLIst
used by RootGraph
"""
import numpy as _np

from rhizoscan.tool import _property    
from rhizoscan.datastructure import Mapping as _Mapping


class GraphList(_Mapping):
    ## del dynamic property when saved (etc...) / no direct access (get_prop)?
    @_property
    def properties(self):
        """ list of list properties (use add_property to add one)"""
        if not hasattr(self,'_property_names'):
            self._property_names = set()
        return list(self._property_names)
    def add_property(self, name, value):
        """
        Add the attribute 'name' with value 'value' to this object, and 'name' to
        this object 'properties' attribute. 
        """
        self.properties# assert existence
        self._property_names.add(name)
        self[name] = value

class NodeList(GraphList):
    def __init__(self,position=None, x=None, y=None):
        """
        Input:   Should give either a position array, or the x & y
            position: a 2xN array of the x,y coordinates of N nodes
            x&y:      2 length N vector array of the x,y coordinates
        """
        if x is not None and y is not None:
            self.position = _np.concatenate((x[None],y[None]), axis=0)
        else:
            self.position = position
            
        ##self.size = self.position.shape[1]-1   ## -1 should be removed ?
        
    @_property
    def x(self):  
        """ x-coordinates of nodes """  
        return self.position[0]
    @_property
    def y(self):
        """ y-coordinates of nodes """  
        return self.position[1]
    
    @_property
    def number(self):  
        """ number of nodes, including dummy (i.e. 0) node """  
        return self.position.shape[1]
        
    def set_segment(self, segment):
        if hasattr(segment,'node'):
            ns = [[] for i in xrange(self.number)]
            for s,sn in enumerate(segment.node[1:]):
                ns[sn[0]].append(s+1)
                ns[sn[1]].append(s+1)
            ns[0] = []
            self.segment = _np.array(ns,dtype=object)
        else:
            self.segment = segment
        
    @_property
    def terminal(self):
        """ bool flag, is node terminal """
        if not hasattr(self, '_terminal'):
            self._terminal = _np.vectorize(len)(self.segment)==1
            self.temporary_attribute.add('_terminal')
        return self._terminal
    
class SegmentList(GraphList):
    ##TODO Segmentlist: add a (private) link to node, and lake length, etc... class properties 
    def __init__(self, node_id, node_list):
        """
        Create a SegmentList from an Nx2 array of nodes pairs
        """
        self.node_list = node_list
        self.node = node_id
        ##self.size = node_id.shape[0]-1  ## -1 should be removed ?!
        
    @_property
    def number(self):  
        """ number of segments, including dummy (i.e. 0) segment """  
        return self.node.shape[0]
        
    @_property
    def length(self):
        """ Compute length of segments from NodeList 'node' """
        if not hasattr(self,'_length'):
            nx = self.node_list.x[self.node]
            ny = self.node_list.y[self.node]
            self._length = ((nx[:,0]-nx[:,1])**2 + (ny[:,0]-ny[:,1])**2)**.5
            self.temporary_attribute.add('_length')
        return self._length
    @length.setter
    def length(self, value):
        self._length = value
        self.clear_temporary_attribute('_length')
    
    @_property
    def direction(self):
        """ Compute direction of segments from NodeList 'node' """
        if not hasattr(self,'_direction'):
            sy = self.node_list.y[self.node]
            sx = self.node_list.x[self.node]
            dsx = _np.diff(sx).ravel()
            dsy = _np.diff(sy).ravel()
            self._direction = _np.arctan2(dsy,dsx)
            self.temporary_attribute.add('_direction')
        return self._direction
    @direction.setter
    def direction(self, value):
        self._direction = value
        self.clear_temporary_attribute('_direction')
        
    @_property
    def terminal(self):
        """ Compute terminal property of segments using attribute node_list """
        if not hasattr(self,'_terminal'):
            self._terminal = _np.any(self.node_list.terminal[self.node],axis=1)
            self.temporary_attribute.add('_terminal')
        return self._terminal
    @terminal.setter
    def terminal(self, value):
        self._terminal = value
        self.clear_temporary_attribute('_terminal')
       
    @_property
    def direction_difference(self):
        """ 
        Array of difference in direction between all segments in List
        
        This difference take into account by which node the segment are connected
        but angle diff for unconnected segment is meaningless
        """
        if not hasattr(self,'_direction_difference'):
            angle = self.direction
            dangle = _np.abs(angle[:,None] - angle[None,:])
            dangle = _np.minimum(dangle, 2*_np.pi-dangle)
            # segments sharing start or end nodes needs to be reverted
            to_revert = _np.any(self.node[:,None,:]==self.node[None,:,:],axis=-1)
            dangle[to_revert] = _np.pi - dangle[to_revert]
            dangle[0,:] = dangle[:,0] = _np.pi
            self._direction_difference = dangle
            self.temporary_attribute.add('_direction_difference')
        return self._direction_difference
    
    @_property
    def distance_to_seed(self):
        """ require the property 'length', 'order' and 'parent' """
        if not hasattr(self,'_distance_to_seed'):
            d2seed = self.length.copy()
            p = self.parent
            for i in self.order:
                d2seed[i] += d2seed[p[i]]
            self.temporary_attribute.add('_distance_to_seed')
        return self._distance_to_seed
        
        
    @property
    def neighbors(self):
        """ 
        Edges array of neighboring segments constructed with `neighbor_array`
        *** It requires the `seed` attribute ***
        """
        if not hasattr(self,'_neighbors'):
            nbor = neighbor_array(self.node_list.segment, self.node, self.seed)
            self._neighbors = nbor
            self.temporary_attribute.add('_neighbors')
        return self._neighbors
    @neighbors.setter
    def neighbors(self, value):
        self.clear_temporary_attribute('_neighbors')
        if value is not None:
            self._neighbors = value
    
    def digraph(self, direction):
        """
        Create the digraph induced by `direction`
        
        `direction` should be a boolean vector array with length equal to the 
        segment number. True value means the segment direction reversed.
        
        Return a neighbor type array such that: 
          - neighbors[...,0] are the  incoming neighbors, and
          - neighbors[...,1] are the outcoming neighbors
        """
        # reverse edge direction
        node = self.node.copy()
        node[direction] = node[direction][...,::-1]
        nbor = neighbor_array(self.node_list.segment, node, self.seed)
            
        # remove edges that are invalid for a directed graph
        # 
        # switch: boolean array with same shape as `nbor` that has True value 
        # where (directed) connection through a neighbors edge requires a change 
        # of one of the segment direction. ie.:
        # 
        # for all edge (i,j) stored in `neighbors`, i.e. j in neighbors[i]: 
        #   - i & j are not in the same relative direction
        #   - i.e. is j a neighbor on side s of i, and i on side s of j ?
        #
        # neighbors that requires switch are invalid in the digraph
        switch = _np.zeros(nbor.shape, dtype=bool)
        sid    = _np.arange(nbor.shape[0])[:,None,None]
        switch[...,0] = (nbor[nbor[...,0],:,0]==sid).any(axis=-1) # side 0
        switch[...,1] = (nbor[nbor[...,1],:,1]==sid).any(axis=-1) # side 1
             
        nbor[switch] = 0
        
        return nbor

class AxeList(GraphList):
    def __init__(self, axes=None, parent=None, plant=None, order=None, segment_list=None, parent_segment='parent'):
        """
        Create an AxeList instance.
        
        :Warning:
            For backward compatibility, it is possible to make an empty AxeList
            without argument, but this will most probably become deprecated
            
        :Inputs:
          - axes:
              A list or array, of the **sorted** list of segment id each 
              root axe contains. The first list (i.e. the 1st axe) should be
              empty: a *dummy* axe.
              This value is stored in this AxeList `segment` attribute
              The list should be in a decreasing priority order (see notes)
          - parent:
              id of the parent axe, as an array of same length as `axes`
          - order:
              An array-like of the order of each axe. Same length as `axe`.
          - plant:
              An array-like of the plant id for each axe. Same length as `axe`.
          - segment_list:
              The SegmentList instance on which this AxeList is constructed.
          - parent_segment:
              The list of the parent segment of all axes. If a string is given, 
              use the the attribute with the name of `segment_list` to infer it.
              See notes.
              
        :Notes:
            The AxeList constructor compute the "main axe" of each segment from
            `segment_list` base on the `order` argument then on the order of 
            appearance in input `axe` list.
            The array containing the id of the selected main axe for each 
            segment is stored in the attribute `segment_axe`.

            It is considered that the parent axe of an axe `a` is the main axe of
            the parent segment of the 1st segment of axe `a`.
        """
        if axes is None: 
            DeprecationWarning("AxeList constructor without argument is deprecated") ##
            return
        
        self.segment = _np.asarray(axes)
        self.order   = _np.asarray(order)
        self.plant   = _np.asarray(plant)
        
        self._segment_list = segment_list
        
        if isinstance(parent_segment, basestring):
            self.parent_segment = segment_list[parent_segment][self.first_segment]
        else:
            self.parent_segment = _np.asarray(parent_segment)
            
        if order is not None:
            self._order = order
            #see property 'order' 
        
        # find id of main axe for all segments
        segment_axe  = _np.zeros(segment_list.number, dtype=int)
        axe_priority = _np.argsort(order[1:])+1
        ##todo: add priority by starting position on parent axe
        ##todo: store axe_priority - as partial_order?
        for o in axe_priority[::-1]:
            slist = self.segment[o]
            segment_axe[slist] = o
        self.segment_axe = segment_axe
        
        if parent is None:
            self.parent = self.segment_axe[self.parent_segment]
        else:
            self.parent = _np.asarray(parent)
            
        
    def _update_version(self, verbose=False):
        """ update AxeList from older version """
        # previous auto compute of parent segment (property sparent)
        if not hasattr(self,'parent_segment'):
            if verbose: print 'sparent property to parent_segment attribute'
            sparent = self._segment_parent[self.first_segment]
            self.parent_segment = sparent
            del self._segment_parent
            
        # replace property "parent"
        if not hasattr(self,'parent'):
            if verbose: print 'parent property to parent attribute'
            self.parent = self.segment_axe[self.parent_segment]
            
        # replace 'order' attribute by property
        if self.__dict__.has_key('order'):
            if verbose: print 'order attribute to order property'
            self._order = self.__dict__['order']
            del self.__dict__['order']
              
    @_property
    def number(self):
        """ number of axes, including dummy (i.e. 0) axe """  
        return len(self.segment)
        
    @_property
    def segment_number(self):
        if not hasattr(self,'_segment_number'):
            self._segment_number = _np.vectorize(len)(self.segment)
            self.temporary_attribute.add('_segment_number')
        return self._segment_number
        
    @_property
    def length(self):
        if not hasattr(self,'_length'):
            axlen = _np.vectorize(lambda slist: self._segment_list.length[slist].sum())
            self._length = axlen(self.segment)
            self.temporary_attribute.add('_length')
        return self._length
    @length.setter
    def length(self, value):
        self.clear_temporary_attribute('_length')
        self._length = value
        
    @_property
    def first_segment(self):                    ## change this attrib name?
        """ first segment of axe """
        if not hasattr(self,'_AxeList__segment1'):
            segment1 = _np.array([sl[0] if len(sl) else 0 for sl in self.segment])
            self.__segment1 = segment1
            self.temporary_attribute.add('_AxeList__segment1')
        return self.__segment1
    @_property
    def sparent(self):                    ## change this attrib name?
        """ compute axe parent segment from segment_list parent property 
        DEPRECATED - should be remove
        """
        DeprecationWarning("AxeList.sparent property => should use 'parent_segment' attribute")
        if not hasattr(self,'_AxeList__parent_segment'):
            sparent = self._segment_parent[self.first_segment]
            self.__parent_segment = sparent
            self.temporary_attribute.add('_AxeList__parent_segment')
        return self.__parent_segment
    
    ##@_property
    ##def parent(self):
    ##    """ 
    ##    ids of parent axe
    ##    
    ##    If not provided, contruct one as the 'main' axe of `sparent`
    ##    """
    ##    if not hasattr(self,'_parent'):
    ##        self._parent = self.segment_axe[self.sparent]
    ##        self.temporary_attribute.add('_parent')
    ##    return self._parent
    ##    
    ##@parent.setter
    ##def parent(self, parent_list):
    ##    self._parent = _np.asarray(parent_list)
    ##    self.temporary_attribute.discard('_parent')
    
    @_property
    def order(self):
        """ axe topological order """
        if not hasattr(self,'_order'):
            raise NotImplementedError("order property")
        return self._order
        
    @order.setter
    def order(self, value):
        self.clear_temporary_attribute('_order')
        self._order = value
        
        
    @_property
    def insertion_angle(self):
        """ insertion angle axe """
        if not hasattr(self,'_AxeList__insertion_angle'):
            insertion_angle = self._segment_list.direction_difference[self.first_segment,self.parent_segment]
            self.__insertion_angle = insertion_angle
            self.temporary_attribute.add('_AxeList__insertion_angle')
        return self.__insertion_angle
    
    ##def _compute_length_properties(self):
    ##    # compute the axe length and arc length of segment w.r.t their axe
    ##    arc_length = _np.array([[] for i in xrange(len(self.segment))])
    ##    axe_length = _np.zeros(len(self.segment))
    ##    segment_number = _np.zeros(len(self.segment),dtype=int)
    ##    for i,slist in enumerate(self.segment):
    ##        if len(slist)==0: continue
    ##        slist = _np.asarray(slist)
    ##        arcL  = _np.cumsum(self._segment_list.length[slist])
    ##        arc_length[i] = arcL
    ##        main_axe = self._segment_list.axe[slist]==i         # if axis are overloaping, update
    ##        arc_length[slist[main_axe]] = arcL[main_axe]        # arc length if axe i is the segment "main" axe 
    ##        axe_length[i] = arcL[-1]
    ##        segment_number[i] = len(arcL)
    ##    self.segment.add_property('axelength',arc_length)
        
    def get_node_list(self):
        """
        Return list of axes as a list of node
        and a list of potential invalid axe: 1-segment axes with no parent_segment
        """
        from scipy.sparse import csr_matrix as csr
        from scipy.sparse.csgraph import depth_first_order as dfo
        
        axe_node = []
        invalid  = []
        term_node = self._segment_list.node_list.terminal
        
        for i,seg_list in enumerate(self.segment):
            if len(seg_list)==0: 
                axe_node.append([])
                continue
                
            seg_node = self._segment_list.node[seg_list]
            spnode   = self._segment_list.node[self.parent_segment[i]]
            if seg_node.shape[0]==1:
                snode0   = set(seg_node[0])
                nparent  = snode0.intersection(spnode)
                if len(nparent)<>1:
                    invalid.append(i)
                    seg_node = seg_node.ravel()
                    axe_node.append(seg_node[term_node[seg_node].argsort()])
                else:
                    axe_node.append(_np.array(list(nparent) + list(snode0.difference(nparent))))
            else:
                c = csr((_np.ones(2*seg_node.shape[0],dtype='uint8'),_np.hstack((seg_node[::-1].T,seg_node[:,::-1].T))))
                s = set(seg_node[0]).difference(seg_node[1]).pop()
                order = dfo(c,s, return_predecessors=False) #nparent.pop()
                axe_node.append(order)
                
        return axe_node,invalid

def neighbor_array(node_segment, segment_node, seed=None, output='array'):
    """
    Create an edges array of neighboring segments
    
    :Inputs:
      - node_segment:
          An array of lists of all segment connected to each node
      - segment_node:
          An array of the nodes pairs connected to each segment
      - seed: optional
          If given, it should be a mask that flags 'seed' segments. Nodes that 
          connect only such seed segments are not use for connection.
      - output:
          Either 'array' or 'list'. See `Output` section for details
    
    :Output:
        If `output` is 'list', return a list of a pair of sets: 
        for each "parent" segment, it contains a pair (tuple) of the set of 
        connected "children" segments, one for each side the "parent".
        
        If `output` is 'array', return an array of shape (S,N,2) with S the
        number of ("parent") segments, N the maximum number for "children" 
        neighbors per "parent" segment and 2 for the 2 "parent" sides. 
        Each neighbors[i,:,k] contains the list of the neighboring segments ids 
        on side `k` of segment `i`.
        In order to fill the array, the missing neighbors are set to 0.
    """
    ns   = node_segment.copy()
    if seed is not None:
        invalid_nodes = _np.vectorize(lambda nslist: (seed[nslist]>0).all())(node_segment)
        ns[invalid_nodes] = set()
    ns[0] = set()
    
    # construct nb1 & nb2 the neighbor array of all segments in direction 1 & 2
    nsbor = _np.vectorize(set)(ns)
    snbor = [(s1.difference([i]),s2.difference([i])) for i,(s1,s2) in enumerate(nsbor[segment_node])]

    if output=='array':
        edge_max = max(map(lambda edg:max(len(edg[0]),len(edg[1])),snbor))
        edge = _np.zeros((len(snbor),edge_max,2), dtype='uint32')
        for i,(nb1,nb2) in enumerate(snbor):
            edge[i,:len(nb1),0] = list(nb1)
            edge[i,:len(nb2),1] = list(nb2)
        return edge
        
    else:
        return snbor

