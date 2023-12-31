# %%

# coding: utf-8

# %%
import warnings

import numpy as np

import matplotlib as mpl
import matplotlib.path

import PIL
from PIL import Image,ImageDraw

import cv2

from shapely.geometry import Polygon
from shapely.strtree import STRtree


from pathlib import Path

from .annotation_handlers import get_points_base

# %%


class wsi(dict):
    
    def __init__(self,img_fname=None,annotation_fname=None, mpp=None, img_dims=None):
        self["img_fname"] = img_fname
        self["annotation_fname"] = annotation_fname        
        self["stored_points"] = dict()
        if img_fname is not None:
            None
            
        # if you don't have an image, such as just making binary masks from annotations
        else:
            if mpp is not None:
                self["mpp"] = mpp
            if img_dims is not None:
                self["img_dims"] = [img_dims]   
        
    def get_points(self,colors_to_use,custom_colors=[]):        
        
        stored_points = get_points_base(self,colors_to_use,custom_colors)
            
        # precomputing polygons and STRTree makes later operations faster, but takes a few extra milliseconds. 
        return stored_points['points'], stored_points['map_idx'],stored_points['polygons'] ,stored_points['STRtree']
    
    def get_largest_region(self,points):                
        
        poly_list = [Polygon(point_set) for point_set in points]            
        areas = [poly.area for poly in poly_list]            
        
        return areas.index(max(areas))
        
    def get_coord_at_mpp(self,coordinate,output_mpp,input_mpp=None):
        """Given a dimension or coordinate, returns what that input would be scaled to the given MPP"""
        
        if input_mpp is None:
            input_mpp = self["mpp"]
                   
        coordinate = int(coordinate * input_mpp / output_mpp)
        
        return coordinate
    
    def get_layer_for_mpp(self,desired_mpp,wh=None):
        """Finds the highest-MPP layer with an MPP > desired_mpp, rescales dimensions to match that layer"""
        
        diff_mpps = [float(desired_mpp) - mpp for mpp in self["mpps"]]
        valid_layers = [(index,diff_mpp) for index,diff_mpp in enumerate(diff_mpps) if diff_mpp>=0]
        valid_diff_mpps = [v[1] for v in valid_layers]
        valid_layers= [v[0] for v in valid_layers]
        if len(valid_layers) == 0:
            warn_message = 'Desired_mpp is lower than minimum image MPP of ' + str(min(self["mpps"]))
            warnings.warn(warn_message)
            target_layer = self["mpps"].index(min(self["mpps"])) 
        else:
            target_layer = valid_layers[valid_diff_mpps.index(min(valid_diff_mpps))]
                
        layer_scale = desired_mpp / self["mpps"][target_layer]        
        
        if wh is not None:
            wh = [int(float(dimension) * layer_scale) for dimension in wh]            
        
        return target_layer, layer_scale, wh
            
    def read_region(self,coords,target_layer,wh):
        """Returns an RGB image of the desired region, will use more libraries when implemented, for now just Openslide"""
        img = self["osh"].read_region(coords,target_layer,wh)
        img = np.array(img)[:,:,0:3] # openslide returns an alpha channel
        
        return img        

    def resize_points(self,points,resize_factor):
                
        points = [[(int(p[0] * resize_factor), int(p[1] * resize_factor)) for p in pointSet] for pointSet in points]
        
        return points.copy()
                                                
    def mask_out_tile(self,desired_mpp,coords,wh,colors_to_use=None,annotation_idx=None,custom_colors=[],point_dict=None,wh_at_base=False):
        """Returns the mask of a tile"""
    
        if wh_at_base:   
            base_wh = wh
            wh = tuple([self.get_coord_at_mpp(dimension,output_mpp=desired_mpp) for dimension in wh])
        else:
            base_wh = tuple([self.get_coord_at_mpp(dimension,output_mpp=self['mpp'],input_mpp=desired_mpp) for dimension in wh])
    
        if point_dict:
            points = point_dict['points']
            map_idx = point_dict['map_idx']
            point_polys = []
            point_tree = []
        else:
            points, map_idx,point_polys,point_tree = self.get_points(colors_to_use,custom_colors)            

        resize_factor = self["mpp"] / desired_mpp                
                
        if type(annotation_idx) == str and annotation_idx.lower() == 'largest':
            largest_idx = self.get_largest_region(points)
            points = [points[largest_idx]]
        elif annotation_idx is not None:
            points = [points[annotation_idx]]
            
        tile_poly = Polygon([(coords[0],coords[1]),(coords[0],coords[1]+base_wh[1]),(coords[0]+base_wh[0],coords[1]+base_wh[1]),(coords[0]+base_wh[0],coords[1])])
        
        mask = np.zeros((wh[1],wh[0]),dtype=np.uint8)                
        
#         points_maps = [point_map for point_map in zip(points,map_idx) if tile_poly.intersects(Polygon(point_map[0]))]
#         points_maps = [point_map for point_map in zip(points,map_idx) if tile_poly.intersects(Polygon(point_map[0]))]
        
        if not point_polys:
            point_polys = [Polygon(point) for point in points]
            point_tree = STRtree(point_polys)
            
        index_by_id = dict((id(pt), i) for i, pt in enumerate(point_polys))
        intersecting_points = [index_by_id[id(pt)] for pt in point_tree.query(tile_poly)]
        
        points_maps = [point_map for idx,point_map in enumerate(zip(points,map_idx)) if idx in intersecting_points]
        if points_maps:
            points,map_idx = zip(*points_maps)        

            # this rounding may de-align the mask and RGB image
            points = self.resize_points(points,resize_factor)

            coords = tuple([int(c * resize_factor) for c in coords])        

            points = [[(int(p[0] - coords[0]), int(p[1] - coords[1])) for p in pointSet] for pointSet in points]

            for annCount, pointSet in enumerate(points):                    
                cv2.fillPoly(mask,[np.asarray(pointSet).reshape((-1,1,2))],map_idx[annCount])
        
        return mask

    def mask_out_annotation(self,desired_mpp=None,colors_to_use=None,custom_colors=[]):        
        """Returns the mask of annotations. Annotations to be returned specified in colors_to_use. Which annotations are on top controlled by order of strings in colors_to_use"""
        
        if desired_mpp is None:
            desired_mpp = self['mpp']
        
        #wh = [self.get_coord_at_mpp(c,output_mpp=desired_mpp,input_mpp=self['mpp']) for c in [self['img_dims'][0][0],self['img_dims'][0][1]]]
        wh = self["wh"]
        return self.mask_out_tile(desired_mpp,(0,0),wh,colors_to_use,None,custom_colors)        
        
    def get_coords_scn(self,coords,scn_wh):
        
        coords = (coords[1] + self["offsets"][1],-coords[0] + self["offsets"][0] - scn_wh[1])
        
        return coords
    
    def get_tile(self,desired_mpp,coords,wh,wh_at_base=False):        
        """Returns the RGB image of a tile. coords are at base MPP, wh is at desired_mpp unless wh_at_base=True, in which case wh is at base"""
        
        if wh_at_base:
            scn_wh = (wh[1],wh[0])
            wh = tuple([self.get_coord_at_mpp(dimension,output_mpp=desired_mpp) for dimension in wh])
                                    
        target_layer, _, scaled_wh = self.get_layer_for_mpp(desired_mpp,wh)
        
        if(Path(self["img_fname"]).suffix == '.scn'):

            # .scn images reads...backwards
            scaled_wh = (scaled_wh[1],scaled_wh[0])
            wh = (wh[1],wh[0])

            if not wh_at_base:
                scn_wh = tuple([self.get_coord_at_mpp(dimension,output_mpp=self["mpp"],input_mpp=desired_mpp) for dimension in wh])
            
            coords = self.get_coords_scn(coords,scn_wh)            
        
        img = self.read_region(coords,target_layer,scaled_wh)
        img = np.array(img)
        
#         interp_method=cv2.INTER_CUIBC
        
        img = cv2.resize(img,wh,interpolation=cv2.INTER_CUBIC)                
        
        if(len(self["img_fname"]) >= 3 and self["img_fname"][-3:] == 'scn'):
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

            
        return img
    
    def get_wsi(self,desired_mpp):
        """Returns the whole image"""                         
        wsi_image = self.get_tile(desired_mpp=desired_mpp,coords=(0,0),wh=self["img_dims"][0],wh_at_base=True)
        
        return wsi_image

    def show_tile_location(self,desired_mpp,coords,wh,wsi_mpp=8):            
        """Returns the whole image with a box showing where the tile of the given inputs would be located"""
        
        target_layer, layer_scale, scaled_wh = self.get_layer_for_mpp(desired_mpp,wh)
        wsi_target_layer, wsi_layer_scale, wsi_scaled_wh = self.get_layer_for_mpp(wsi_mpp,wh)
                
        rect_coords = tuple([self.get_coord_at_mpp(c,wsi_mpp) for c in coords])        
        wsi_scaled_wh = tuple([self.get_coord_at_mpp(dimension,wsi_mpp,input_mpp=desired_mpp) for dimension in wh])        
        
        wsi_image = self.get_wsi(wsi_mpp).copy()        

        cv2.rectangle(wsi_image,rect_coords,tuple(map(lambda x,y: x+y,rect_coords,wsi_scaled_wh)),(0,255,0),int(np.max(np.shape(wsi_image))/200))

        return wsi_image
    
    def get_dimensions_of_annotation(self,colors_to_use,annotation_idx,custom_colors=[]):
        points, _,_,_ = self.get_points(colors_to_use,custom_colors)
        
        if(not points):
            print('No annotations of selected color')
            bounding_box = None
        else:
        
            poly_list = [Polygon(point_set) for point_set in points]


            if type(annotation_idx) == str and annotation_idx.lower() == 'largest':
                areas = [poly.area for poly in poly_list]
                annotation_idx = areas.index(max(areas))

            bounding_box = [int(c) for c in poly_list[annotation_idx].bounds]

        return bounding_box
    
    def get_annotated_region(self,desired_mpp,colors_to_use,annotation_idx,mask_out_roi=True,tile_coords=None,tile_wh=None,wh_add=(0,0),return_img=True,custom_colors=[],restrict_to_anno=True,all_annos=False):
        """Returns an RGB image of the specified annotated region."""
            
        points, map_idx,_, _ = self.get_points(colors_to_use,custom_colors)
        
        if(not points):
            print('No annotations of selected color')
            img = None
            mask = None
        else:        

            if type(annotation_idx) == str and annotation_idx.lower() == 'largest':
                poly_list = [Polygon(point_set) for point_set in points]
                areas = [poly.area for poly in poly_list]
                annotation_idx = areas.index(max(areas))
                bounding_box = poly_list[annotation_idx].bounds
            else:
                bounding_box = Polygon(points[annotation_idx]).bounds

            point_dict = dict()
            point_dict['points'] = [points[annotation_idx]]
            point_dict['map_idx'] = [map_idx[annotation_idx]]                                    

            coords = tuple([int(bounding_box[0]),int(bounding_box[1])])
            wh = tuple([int(bounding_box[2]-bounding_box[0]),int(bounding_box[3]-bounding_box[1])])
                             
            if(tile_coords):
                coords = tuple([coords[0]+tile_coords[0],coords[1]+tile_coords[1]])
                
            if(tile_wh):
                wh = tile_wh
                
            if(restrict_to_anno and tile_wh):
                if(coords[0]+tile_wh[0] > bounding_box[2]):
                    tile_wh[0] = bounding_box[2] - coords[0]

                if(coords[1]+tile_wh[1] > bounding_box[3]):
                    tile_wh[1] = bounding_box[3] - coords[1]
                
            
            wh = [wh[k]+wh_add[k] for k in [0,1]]
                        
            if(return_img):
                img = self.get_tile(desired_mpp,coords,wh,wh_at_base=True)
                img = np.asarray(img)
            else:
                img = None

            wh = [self.get_coord_at_mpp(dimension,output_mpp=desired_mpp) for dimension in wh]            
            
            point_dict = None if all_annos else point_dict
            mask = self.mask_out_tile(desired_mpp,coords,wh,colors_to_use=colors_to_use,point_dict=point_dict)
                        
            if(mask_out_roi and return_img):
                img = cv2.bitwise_and(img,img,mask=np.uint8(mask))

        return img, mask
