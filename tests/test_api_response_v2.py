"""Deterministic contract fixtures, NOT real model inference or quality tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class ResponseV2Tests(unittest.TestCase):
    def fields(self,classification=None,segmentation=None,roi=None):
        from woundcare_contract import build_response_v2_fields
        return build_response_v2_fields(classification_result=classification,segmentation_result=segmentation,
            original_shape=(100,120),classification_threshold=.60,
            classification_model_identity={'file':r'C:\private\classifier.pt','sha256':'a'*64,'task':'classify'},
            segmentation_model_identity={'file':r'C:\private\segmenter.pt','sha256':'b'*64,'task':'segment'},
            merged_roi=roi,review_required=True,classification_source='full_image')

    def test_seven_scores_preserve_actual_result_order_and_values(self):
        from woundcare_contract import build_response_v2_fields
        import numpy as np
        from ultralytics.engine.results import Results
        names={0:'Abrasions',1:'Bruises',2:'Burns',3:'Cut',4:'Ingrown_nails',5:'Laceration',6:'Stab_wound'}
        result=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',names,
            probs=np.array([.70,.10,.05,.05,.04,.03,.03],dtype=np.float64))
        out=build_response_v2_fields(classification_result=result,segmentation_result=None,
            original_shape=(100,120),classification_threshold=.60,
            classification_model_identity={'file':'best.pt','sha256':'a'*64,'task':'classify'},
            segmentation_model_identity={'file':'best.pt','sha256':'b'*64,'task':'segment'},
            merged_roi=None,review_required=True,classification_source='full_image')
        self.assertEqual(out['schema_version'],'2.0.0')
        self.assertEqual([r['score'] for r in out['seven_class_scores']],[.70,.10,.05,.05,.04,.03,.03])
        self.assertEqual(out['top1_index'],0)
        self.assertEqual(out['top1_name'],'Abrasions')
        self.assertEqual(out['confidence_gate_status'],'ABOVE_THRESHOLD')
        self.assertTrue(out['review_required'])
        self.assertEqual(out['care_mapping_status'],'NOT_CONFIGURED')
        self.assertEqual(out['measurement_status'],'NOT_VALIDATED_NO_CALIBRATED_SCALE')

    def test_two_true_masks_are_not_one_merged_roi_or_boxes(self):
        import numpy as np
        import torch
        from ultralytics.engine.results import Results
        masks=np.zeros((2,20,24),dtype=np.uint8)
        masks[0,2:6,3:9]=1
        masks[1,10:16,12:20]=1
        result=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',{0:'Wound'},
            boxes=np.array([[15,10,45,30,.9,0],[60,50,100,80,.8,0]],dtype=np.float32),masks=torch.from_numpy(masks))
        out=self.fields(segmentation=result,roi={'x1':5,'y1':5,'x2':110,'y2':90})
        self.assertEqual(out['segmentation_instance_count'],2)
        self.assertEqual(len(out['predicted_bounding_boxes']),2)
        self.assertEqual(out['merged_roi']['x2'],110)
        self.assertEqual(out['original_image_size'],{'width':120,'height':100})
        self.assertEqual(out['polygon_coordinate_system'],'ORIGINAL_IMAGE_XY_PIXELS')
        for i,instance in enumerate(out['segmentation_instances']):
            self.assertEqual(instance['mask']['size'],[20,24])
            decoded=[]; bit=0
            for count in instance['mask']['counts']:
                decoded.extend([bit]*count); bit=1-bit
            self.assertEqual(decoded,masks[i].reshape(-1).tolist())
            self.assertTrue(instance['polygon'])
            self.assertTrue(all(0<=x<=120 and 0<=y<=100 for x,y in instance['polygon']))

    def test_schema_defines_v1_and_v2_required_fields(self):
        from woundcare_contract import PredictResponseV2, ResponseV2Fields
        schema=PredictResponseV2.model_json_schema()
        for key in ['image','detections','classes','llm_advice','requires_human_review',
                    'schema_version','seven_class_scores','segmentation_instances','measurement_status']:
            self.assertIn(key,schema['required'])
        value=self.fields()
        self.assertEqual(ResponseV2Fields.model_validate(value).model_dump(mode='json'),value)

    def test_result_observer_keeps_frozen_v1_outputs_and_model_inputs(self):
        import numpy as np
        from ultralytics.engine.results import Results
        from woundcare_inference import CascadeConfig,infer_segmentation_cascade
        image=np.zeros((100,120,3),dtype=np.uint8)
        seg=Results(image,'synthetic-unit',{0:'Wound'},boxes=np.empty((0,6),dtype=np.float32))
        cls=Results(image,'synthetic-unit',{0:'Abrasions',1:'Bruises',2:'Burns',3:'Cut',4:'Ingrown_nails',5:'Laceration',6:'Stab_wound'},
            probs=np.array([.70,.10,.05,.05,.04,.03,.03]))
        class FixedBoundaryModel:
            names=cls.names
            def __init__(self,result): self.result=result; self.inputs=[]
            def predict(self,**kwargs): self.inputs.append(kwargs); return [self.result]
        segmenter,classifier=FixedBoundaryModel(seg),FixedBoundaryModel(cls)
        captured={}
        observed=infer_segmentation_cascade(image,segmenter,classifier,CascadeConfig(),
            result_observer=lambda kind,result:captured.__setitem__(kind,result))
        self.assertIs(captured['classification'],cls)
        self.assertIs(captured['segmentation'],seg)
        # Frozen V1 fixture: adding the observer must not change the public
        # inference dictionary or either model's predict parameters. Keeping
        # the fixture inline makes this regression test runnable from a clean
        # public clone without the private deployment evidence directory.
        self.assertEqual(observed,{
            'mode':'full_image_primary',
            'fallback_used':False,
            'detected_boxes':0,
            'segment_instances':0,
            'segment_confidence':None,
            'crop_box':None,
            'label':'Abrasions',
            'class_confidence':0.7,
            'low_confidence':False,
            'classification_source':'full_image',
        })
        self.assertEqual(len(segmenter.inputs),1)
        self.assertIs(segmenter.inputs[0]['source'],image)
        self.assertEqual(
            {k:v for k,v in segmenter.inputs[0].items() if k!='source'},
            {'conf':0.25,'iou':0.7,'imgsz':224,'device':'cpu','verbose':False},
        )
        self.assertEqual(len(classifier.inputs),1)
        self.assertIs(classifier.inputs[0]['source'],image)
        self.assertEqual(
            {k:v for k,v in classifier.inputs[0].items() if k!='source'},
            {'imgsz':224,'device':'cpu','verbose':False},
        )

    def test_no_results_are_explicitly_unavailable_without_fabricated_scores(self):
        out=self.fields()
        self.assertIsNone(out['seven_class_scores'])
        self.assertIsNone(out['top1_index'])
        self.assertEqual(out['confidence_gate_status'],'UNAVAILABLE')
        self.assertEqual(out['segmentation_instance_count'],0)
        self.assertEqual(out['segmentation_instances'],[])
        self.assertEqual(out['predicted_bounding_boxes'],[])
        self.assertIsNone(out['merged_roi'])

    def test_boxes_without_masks_are_not_segmentation_instances(self):
        import numpy as np
        from ultralytics.engine.results import Results
        r=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',{0:'Wound'},
            boxes=np.array([[0,0,40,40,.8,0]],dtype=np.float32))
        out=self.fields(segmentation=r,roi={'x1':0,'y1':0,'x2':45,'y2':45})
        self.assertEqual(out['segmentation_instance_count'],0)
        self.assertEqual(out['segmentation_instances'],[])
        self.assertEqual(len(out['predicted_bounding_boxes']),1)
        self.assertEqual(out['segmentation_status'],'NO_MASK')

    def test_bad_probabilities_are_not_normalized_or_invented(self):
        import numpy as np
        from ultralytics.engine.results import Results
        names={0:'Abrasions',1:'Bruises',2:'Burns',3:'Cut',4:'Ingrown_nails',5:'Laceration',6:'Stab_wound'}
        for values in [[.7,.1,float('nan'),.05,.04,.03,.03],[.7]*7,[-.1,.5,.2,.1,.1,.1,.1]]:
            with self.subTest(values=str(values)):
                r=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',names,probs=np.array(values))
                out=self.fields(classification=r)
                self.assertEqual(out['confidence_gate_status'],'INVALID_RESULT')
                self.assertIsNone(out['seven_class_scores'])
                self.assertTrue(out['review_required'])

    def test_model_mapping_not_an_assumed_alphabetical_order(self):
        import numpy as np
        from ultralytics.engine.results import Results
        names={0:'Stab_wound',1:'Bruises',2:'Burns',3:'Cut',4:'Ingrown_nails',5:'Laceration',6:'Abrasions'}
        r=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',names,probs=np.array([.7,.1,.05,.05,.04,.03,.03]))
        out=self.fields(classification=r)
        self.assertEqual(out['top1_name'],'Stab_wound')
        self.assertEqual(out['seven_class_scores'][0]['name'],'Stab_wound')

    def test_confidence_boundary_never_proves_diagnosis_or_ood(self):
        import numpy as np
        from ultralytics.engine.results import Results
        names={0:'Abrasions',1:'Bruises',2:'Burns',3:'Cut',4:'Ingrown_nails',5:'Laceration',6:'Stab_wound'}
        for first,expected in [(.59,'BELOW_THRESHOLD'),(.60,'ABOVE_THRESHOLD'),(.99,'ABOVE_THRESHOLD')]:
            r=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',names,
                probs=np.array([first,1-first,0,0,0,0,0],dtype=np.float64))
            out=self.fields(classification=r)
            self.assertEqual(out['confidence_gate_status'],expected)
            self.assertTrue(out['review_required'])
            self.assertEqual(out['full_image_classification']['ood_rejection_status'],'NOT_VALIDATED')
            self.assertEqual(out['care_mapping_status'],'NOT_CONFIGURED')

    def test_identity_strips_absolute_paths_and_arbitrary_private_metadata(self):
        out=self.fields()
        self.assertEqual(out['classification_model_identity']['file'],'classifier.pt')
        self.assertEqual(out['segmentation_model_identity']['file'],'segmenter.pt')
        import json
        self.assertNotIn('C:',json.dumps(out))

    def test_rle_preserves_holes_and_disconnected_native_pixels(self):
        import numpy as np
        import torch
        from ultralytics.engine.results import Results
        masks=np.zeros((1,20,24),dtype=np.uint8)
        masks[0,2:10,2:10]=1; masks[0,4:7,4:7]=0; masks[0,14:18,18:22]=1
        r=Results(np.zeros((100,120,3),dtype=np.uint8),'synthetic-unit',{0:'Wound'},
            boxes=np.array([[0,0,120,100,.8,0]],dtype=np.float32),masks=torch.from_numpy(masks))
        out=self.fields(segmentation=r)
        mask=out['segmentation_instances'][0]['mask']; bits=[]; bit=0
        for count in mask['counts']: bits.extend([bit]*count); bit=1-bit
        self.assertEqual(bits,masks[0].reshape(-1).tolist())
        self.assertIn('NOT_LOSSLESS',out['polygon_representation'])

    def test_invalid_polygon_is_null_not_clipped_or_replaced_with_roi(self):
        import numpy as np
        from types import SimpleNamespace as DTO
        result=DTO(boxes=None,masks=DTO(data=np.ones((1,2,3),dtype=np.uint8),xy=[np.array([[0,0],[121,0],[120,100]])]))
        out=self.fields(segmentation=result,roi={'x1':0,'y1':0,'x2':120,'y2':100})
        self.assertEqual(out['segmentation_instance_count'],1)
        self.assertIsNone(out['segmentation_instances'][0]['polygon'])
        self.assertIsNone(out['segmentation_instances'][0]['predicted_box_index'])

    def test_nonbinary_mask_is_not_silently_thresholded(self):
        import numpy as np
        from types import SimpleNamespace as DTO
        r=DTO(boxes=None,masks=DTO(data=np.full((1,2,3),.5),xy=[]))
        out=self.fields(segmentation=r)
        self.assertIsNone(out['segmentation_instances'][0]['mask'])
        self.assertEqual(out['segmentation_instance_count'],1)

    def test_schema_rejects_auto_approval_and_invented_configured_care(self):
        from pydantic import ValidationError
        from woundcare_contract import ResponseV2Fields
        for change in [{'review_required':False},{'care_mapping_status':'APPROVED'},{'measurement_status':'VALIDATED'}]:
            value=self.fields(); value.update(change)
            with self.assertRaises(ValidationError): ResponseV2Fields.model_validate(value)


if __name__=='__main__': unittest.main()
