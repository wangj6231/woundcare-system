"""Public response serialization only: never infer, normalize scores or approve care."""
from __future__ import annotations

import math
from pathlib import PureWindowsPath
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)


class ClassScore(ContractModel):
    index: int=Field(ge=0,le=6)
    name: str
    score: float=Field(ge=0,le=1)


class PublicModelIdentity(ContractModel):
    file: str|None
    sha256: str|None=Field(pattern=r'^[0-9a-f]{64}$')
    task: str|None
    input_size: int|None


class FullImageClassification(ContractModel):
    source: str
    is_primary: bool
    status: Literal['AVAILABLE','UNAVAILABLE','INVALID_RESULT']
    raw_top1_confidence: float|None=Field(ge=0,le=1)
    ood_rejection_status: Literal['NOT_VALIDATED']


class MaskRLE(ContractModel):
    encoding: Literal['BINARY_RLE_C_ORDER']
    size: list[Annotated[int,Field(gt=0)]]=Field(min_length=2,max_length=2)
    counts: list[Annotated[int,Field(ge=0)]]
    starts_with: Literal[0]
    coordinate_system: Literal['NATIVE_RESULT_MASK_GRID']

    @model_validator(mode='after')
    def complete_grid(self):
        if sum(self.counts)!=self.size[0]*self.size[1]:
            raise ValueError('RLE must represent the complete native grid')
        return self


class SegmentationInstance(ContractModel):
    index: int=Field(ge=0)
    mask: MaskRLE|None
    polygon: list[list[float]]|None
    predicted_box_index: int|None=Field(ge=0)


class PredictedBox(ContractModel):
    index: int=Field(ge=0)
    xyxy: list[float]|None=Field(min_length=4,max_length=4)
    confidence: float|None=Field(ge=0,le=1)
    class_index: int|None=Field(ge=0)
    coordinate_system: Literal['ORIGINAL_IMAGE_XYXY_PIXEL_EDGES']


class ImageSize(ContractModel):
    width: int=Field(gt=0)
    height: int=Field(gt=0)


class ROIBox(ContractModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ResponseV2Fields(ContractModel):
    schema_version: Literal['2.0.0']
    full_image_classification: FullImageClassification
    seven_class_scores: list[ClassScore]|None=Field(min_length=7,max_length=7)
    top1_index: int|None=Field(ge=0,le=6)
    top1_name: str|None
    classification_threshold: float=Field(ge=0,le=1)
    confidence_gate_status: Literal['ABOVE_THRESHOLD','BELOW_THRESHOLD','UNAVAILABLE','INVALID_RESULT']
    segmentation_model_identity: PublicModelIdentity
    classification_model_identity: PublicModelIdentity
    segmentation_instance_count: int=Field(ge=0)
    segmentation_instances: list[SegmentationInstance]
    predicted_bounding_boxes: list[PredictedBox]
    segmentation_status: Literal['UNAVAILABLE','NO_MASK','INVALID_MASK_SHAPE','INCONSISTENT_MASK_BOX_COUNTS','AVAILABLE_WITH_MISSING_REPRESENTATION','AVAILABLE']
    original_image_size: ImageSize
    polygon_coordinate_system: Literal['ORIGINAL_IMAGE_XY_PIXELS']
    polygon_representation: Literal['ULTRALYTICS_NATIVE_CONTOUR_NOT_LOSSLESS_FOR_HOLES_OR_DISCONNECTED_REGIONS']
    merged_roi: ROIBox|None
    merged_roi_kind: Literal['PADDED_UNION_DISPLAY_ROI_NOT_INSTANCE']
    review_required: Literal[True]
    care_mapping_status: Literal['NOT_CONFIGURED']
    measurement_status: Literal['NOT_VALIDATED_NO_CALIBRATED_SCALE']

    @model_validator(mode='after')
    def matching_counts_and_coordinates(self):
        if self.segmentation_instance_count!=len(self.segmentation_instances):
            raise ValueError('Instance count must count actual Result masks, not boxes or ROI')
        for instance in self.segmentation_instances:
            if instance.polygon is not None:
                if len(instance.polygon)<3 or any(len(p)!=2 or not 0<=p[0]<=self.original_image_size.width or not 0<=p[1]<=self.original_image_size.height for p in instance.polygon):
                    raise ValueError('Polygon outside original image edge coordinates')
        return self


class LegacyDetection(ContractModel):
    box: ROIBox
    confidence: float|None
    label: str
    class_confidence: float|None
    source: str


class LegacyClassifierIdentity(ContractModel):
    file: str
    sha256: str|None
    input_size: int
    inference_mode: str


class PredictResponseV2(ResponseV2Fields):
    """Additive V2 contract: every existing V1 response field remains required."""
    image: str
    detected_boxes: int
    detections: list[LegacyDetection]
    classes: list[str]
    inference_mode: str
    fallback_used: bool
    segment_confidence: float|None
    class_confidence: float|None
    crop_box: ROIBox|None
    low_confidence: bool
    llm_advice: str
    advice_mode: str
    model_weights_learn_from_feedback: bool
    rag_guidance: list[dict[str,Any]]
    requires_human_review: Literal[True]
    localization_available: bool
    advice_blocked: bool
    review_reasons: list[str]
    model: LegacyClassifierIdentity


CLASS_NAMES={'Abrasions','Bruises','Burns','Cut','Ingrown_nails','Laceration','Stab_wound'}


def _classification(result: Any, threshold: float) -> dict[str, Any]:
    out={'seven_class_scores':None,'top1_index':None,'top1_name':None,
         'confidence_gate_status':'UNAVAILABLE','raw_top1_confidence':None}
    probabilities=getattr(result,'probs',None)
    if probabilities is None:
        return out
    try:
        values=probabilities.data
        if hasattr(values,'detach'): values=values.detach().cpu()
        scores=values.tolist()
        names=result.names
        labels=[str(names[i]) for i in range(7)]
        assert len(scores)==7 and set(labels)==CLASS_NAMES
        assert all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in scores)
        assert abs(sum(scores)-1)<=1e-4
        index=int(probabilities.top1)
        confidence=float(probabilities.top1conf)
        assert 0<=index<7 and confidence==scores[index] and confidence==max(scores)
    except (AttributeError,TypeError,ValueError,IndexError,KeyError,AssertionError):
        out['confidence_gate_status']='INVALID_RESULT'
        return out
    out.update(seven_class_scores=[{'index':i,'name':labels[i],'score':v} for i,v in enumerate(scores)],
        top1_index=index,top1_name=labels[index],raw_top1_confidence=confidence,
        confidence_gate_status='ABOVE_THRESHOLD' if confidence>=threshold else 'BELOW_THRESHOLD')
    return out


def _public_identity(identity: dict[str,Any]) -> dict[str,Any]:
    # Never propagate arbitrary source metadata or absolute checkpoint paths.
    return {'file':PureWindowsPath(identity.get('file') or '').name or None,
        'sha256':identity.get('sha256'),'task':identity.get('task'),
        'input_size':identity.get('input_size')}


def _array(value: Any) -> np.ndarray:
    if hasattr(value,'detach'): value=value.detach().cpu().numpy()
    return np.asarray(value)


def _mask_rle(mask: np.ndarray) -> dict[str,Any]|None:
    """Lossless serialization of a native binary mask; no threshold or resampling."""
    if mask.ndim!=2 or not np.isfinite(mask).all() or not np.isin(mask,[0,1]).all():
        return None
    flat=mask.reshape(-1)
    if not len(flat): return None
    edges=np.r_[0,np.flatnonzero(flat[1:]!=flat[:-1])+1,len(flat)]
    counts=np.diff(edges).tolist()
    if flat[0]==1: counts.insert(0,0)
    return {'encoding':'BINARY_RLE_C_ORDER','size':[int(v) for v in mask.shape],
        'counts':counts,'starts_with':0,'coordinate_system':'NATIVE_RESULT_MASK_GRID'}


def _segmentation(result: Any,height: int,width: int) -> dict[str,Any]:
    out={'segmentation_instance_count':0,'segmentation_instances':[],
        'predicted_bounding_boxes':[],'segmentation_status':'UNAVAILABLE' if result is None else 'NO_MASK'}
    if result is None: return out
    boxes=getattr(result,'boxes',None)
    if boxes is not None:
        coords=_array(boxes.xyxy); conf=_array(boxes.conf); classes=_array(boxes.cls)
        for i,xyxy in enumerate(coords):
            valid=len(xyxy)==4 and np.isfinite(xyxy).all() and 0<=xyxy[0]<=xyxy[2]<=width and 0<=xyxy[1]<=xyxy[3]<=height
            confidence=float(conf[i]); class_index=float(classes[i])
            out['predicted_bounding_boxes'].append({'index':i,'xyxy':xyxy.tolist() if valid else None,
                'confidence':confidence if math.isfinite(confidence) and 0<=confidence<=1 else None,
                'class_index':int(class_index) if math.isfinite(class_index) and class_index.is_integer() else None,
                'coordinate_system':'ORIGINAL_IMAGE_XYXY_PIXEL_EDGES'})
    masks=getattr(result,'masks',None)
    if masks is None: return out
    data=_array(masks.data)
    if data.ndim!=3:
        out['segmentation_status']='INVALID_MASK_SHAPE'
        return out
    out['segmentation_instance_count']=len(data)
    try: polygons=masks.xy
    except (AttributeError,TypeError,ValueError): polygons=[]
    aligned=len(out['predicted_bounding_boxes'])==len(data)
    issues=[]
    for i,mask in enumerate(data):
        encoded=_mask_rle(mask)
        polygon=None
        if i<len(polygons):
            xy=_array(polygons[i])
            if (xy.ndim==2 and xy.shape[1]==2 and len(xy)>=3 and np.isfinite(xy).all()
                    and (xy[:,0]>=0).all() and (xy[:,0]<=width).all()
                    and (xy[:,1]>=0).all() and (xy[:,1]<=height).all()):
                polygon=xy.tolist()
        if encoded is None: issues.append('NON_BINARY_OR_INVALID_NATIVE_MASK')
        if polygon is None: issues.append('POLYGON_UNAVAILABLE_OR_INVALID')
        out['segmentation_instances'].append({'index':i,'mask':encoded,'polygon':polygon,
            'predicted_box_index':i if aligned else None})
    out['segmentation_status']=('INCONSISTENT_MASK_BOX_COUNTS' if not aligned else
        'AVAILABLE_WITH_MISSING_REPRESENTATION' if issues else 'AVAILABLE') if len(data) else 'NO_MASK'
    return out


def build_response_v2_fields(*,classification_result: Any,segmentation_result: Any,
        original_shape: tuple[int,int],classification_threshold: float,
        classification_model_identity: dict[str,Any],segmentation_model_identity: dict[str,Any],
        merged_roi: dict[str,int]|None,review_required: bool,classification_source: str) -> dict[str,Any]:
    """Serialize already computed Results; no model calls or image transformations."""
    cls=_classification(classification_result,classification_threshold)
    confidence=cls.pop('raw_top1_confidence')
    height,width=original_shape
    seg=_segmentation(segmentation_result,height,width)
    fields={'schema_version':'2.0.0',**cls,
        'full_image_classification':{'source':classification_source,'is_primary':classification_source=='full_image',
            'status':'AVAILABLE' if cls['seven_class_scores'] is not None else cls['confidence_gate_status'],
            'raw_top1_confidence':confidence,'ood_rejection_status':'NOT_VALIDATED'},
        'classification_threshold':classification_threshold,
        'classification_model_identity':_public_identity(classification_model_identity),
        'segmentation_model_identity':_public_identity(segmentation_model_identity),
        **seg,'original_image_size':{'width':width,'height':height},
        'polygon_coordinate_system':'ORIGINAL_IMAGE_XY_PIXELS',
        'polygon_representation':'ULTRALYTICS_NATIVE_CONTOUR_NOT_LOSSLESS_FOR_HOLES_OR_DISCONNECTED_REGIONS',
        'merged_roi':merged_roi,'merged_roi_kind':'PADDED_UNION_DISPLAY_ROI_NOT_INSTANCE',
        'review_required':bool(review_required),'care_mapping_status':'NOT_CONFIGURED',
        'measurement_status':'NOT_VALIDATED_NO_CALIBRATED_SCALE'}
    return ResponseV2Fields.model_validate(fields).model_dump(mode='json')
