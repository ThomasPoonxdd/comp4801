from deep_sort_realtime.deepsort_tracker import DeepSort

def Deep_Sort_Implement(frame=None):
    
    deepSortTracker = DeepSort(max_age=10)
    bbs = object_detector.detect(frame) 
    tracks = deepSortTracker.update_tracks(bbs, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
    

if __name__ == '__main__':
    Deep_Sort_Implement()