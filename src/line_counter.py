
import numpy as np
from collections import defaultdict

class LineCounter:
    """
    Virtual line-based pedestrian flow counter
    Supports both horizontal and vertical line counting modes
    """
    def __init__(self, line_position=None, direction='horizontal', max_tracks=1000):
        self.max_tracks = max_tracks
        
        # Virtual line position and direction
        self.line_pos = line_position
        self.direction = direction  # 'horizontal' or 'vertical'
        
        # In/out counts
        self.in_count = 0
        self.out_count = 0
        self.total_count = 0
        
        # Real-time count for current frame
        self.current_count = 0
        
        # Track history {track_id: [(x, y), ...]}
        self.track_history = defaultdict(list)
        
        # Set of tracked IDs that have already been counted to avoid double counting
        self.counted_tracks = set()
        
        # Record all seen track_ids 
        self.seen_track_ids = set()

    def update(self, tracks, frame_shape=None):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        frame_shape: (height, width) - image dimensions, used for automatic virtual line positioning
        Update counting logic with virtual line-based flow counting
        """
        self.current_count = len(tracks)
        
        # Automatically set virtual line position (if not specified)
        if self.line_pos is None and frame_shape is not None:
            if self.direction == 'horizontal':
                self.line_pos = frame_shape[0] // 2  # Middle height of the image
            else:
                self.line_pos = frame_shape[1] // 2  # Middle width of the image
        elif self.line_pos is None:
            # If no frame_shape information is available, use default value
            self.line_pos = 360
        
        # Update seen_track_ids
        for track in tracks:
            if len(track) >= 5:
                track_id = track[4]
                if track_id not in self.seen_track_ids:
                    self.seen_track_ids.add(track_id)
        
        # If no virtual line position is set, perform simple counting only
        if self.line_pos is None:
            self.total_count = len(self.seen_track_ids)
            return
        
        # In/out statistics based on virtual line
        current_tracks = {}
        
        # Process tracking results for current frame
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                # Calculate bounding box center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                
                current_tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'center': center
                }
                
                # Update track history
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 30:  # Limit history length
                    self.track_history[track_id].pop(0)
        
        # Check if tracks cross the counting line
        for track_id, track_data in current_tracks.items():
            if track_id in self.counted_tracks:
                continue
                
            history = self.track_history[track_id]
            if len(history) >= 2:
                prev_center = history[-2]
                curr_center = history[-1]
                
                if self._cross_line(prev_center, curr_center):
                    direction = self._get_direction(prev_center, curr_center)
                    if direction == 'in':
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    self.total_count = self.in_count + self.out_count
                    self.counted_tracks.add(track_id)

    def _cross_line(self, p1, p2):
        """Check if the line connecting two points crosses the counting line"""
        if self.direction == 'horizontal':
            # Horizontal line: check if y coordinates cross line_pos
            return (p1[1] < self.line_pos and p2[1] >= self.line_pos) or \
                   (p1[1] > self.line_pos and p2[1] <= self.line_pos)
        else:
            # Vertical line: check if x coordinates cross line_pos
            return (p1[0] < self.line_pos and p2[0] >= self.line_pos) or \
                   (p1[0] > self.line_pos and p2[0] <= self.line_pos)
    
    def _get_direction(self, p1, p2):
        """Determine movement direction"""
        if self.direction == 'horizontal':
            # Horizontal line: downward movement is 'in', upward movement is 'out'
            return 'in' if p2[1] > p1[1] else 'out'
        else:
            # Vertical line: rightward movement is 'in', leftward movement is 'out'
            return 'in' if p2[0] > p1[0] else 'out'

    def get_counts(self):
        """
        Return cumulative total count and real-time total count
        Returns: (current_count, total_count, in_count, out_count)
        """
        return self.current_count, self.total_count, self.in_count, self.out_count
    
    def reset(self):
        """Reset counter"""
        self.in_count = 0
        self.out_count = 0
        self.total_count = 0
        self.current_count = 0
        self.track_history.clear()
        self.counted_tracks.clear()
        self.seen_track_ids.clear()
    
    def get_line_info(self):
        """Get counting line information"""
        return {
            'position': self.line_pos,
            'direction': self.direction
        }