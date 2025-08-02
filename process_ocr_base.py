import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import subprocess
import json

from util_func import width, height, scale, feature_dim, sigmoid, decode_ruby, modulo_list, softmax, calc_predid
from const import encoder_add_dim, max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT, decoder_MSK, decoder_PAD
encoder_dim = feature_dim + encoder_add_dim

UNICODE_WHITESPACE_CHARACTERS = [
    "\u0009", # character tabulation
    "\u000a", # line feed
    "\u000b", # line tabulation
    "\u000c", # form feed
    "\u000d", # carriage return
    "\u0020", # space
    "\u0085", # next line
    "\u00a0", # no-break space
    "\u1680", # ogham space mark
    "\u2000", # en quad
    "\u2001", # em quad
    "\u2002", # en space
    "\u2003", # em space
    "\u2004", # three-per-em space
    "\u2005", # four-per-em space
    "\u2006", # six-per-em space
    "\u2007", # figure space
    "\u2008", # punctuation space
    "\u2009", # thin space
    "\u200A", # hair space
    "\u2028", # line separator
    "\u2029", # paragraph separator
    "\u202f", # narrow no-break space
    "\u205f", # medium mathematical space
    "\u3000", # ideographic space
]

class OCR_Processer(ABC):
    def __init__(self, step_ratio=0.6, cut_off=0.4):
        super().__init__()

        self.step_ratio = step_ratio
        self.stepx = int(width * self.step_ratio)
        self.stepy = int(height * self.step_ratio)

        self.cut_off = cut_off

    @abstractmethod
    def call_detector(self, image_input):
        raise NotImplementedError()

    @abstractmethod
    def call_transformer(self, encoder_input):
        raise NotImplementedError()

    def call_OCR(self, target_file, resize=1.0):
        im0 = Image.open(target_file).convert('RGB')
        if resize != 1.0:
            im0 = im0.resize((int(im0.width * resize), int(im0.height * resize)), resample=Image.Resampling.BILINEAR)
        im0 = np.asarray(im0)

        padx = max(0, (width - im0.shape[1]) % self.stepx, width - im0.shape[1])
        pady = max(0, (height - im0.shape[0]) % self.stepy, height - im0.shape[0])
        im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

        im = im0.astype(np.float32)

        ds0 = []
        for y in range(0, im0.shape[0] - height + 1, self.stepy):
            for x in range(0, im0.shape[1] - width + 1, self.stepx):
                ds0.append({
                    'input': np.expand_dims(im[y:y+height,x:x+width,:], 0),
                    'offsetx': x,
                    'offsety': y,
                })

        locations, glyphfeatures, lines, seps = self.run_detector(ds0, im)

        h, w = lines.shape
        input_binary = int(0).to_bytes(4, 'little')
        input_binary += int(w).to_bytes(4, 'little')
        input_binary += int(h).to_bytes(4, 'little')
        input_binary += lines.tobytes()
        input_binary += seps.tobytes()
        input_binary += int(locations.shape[0]).to_bytes(4, 'little')
        input_binary += locations[:,1:].tobytes()

        print('run')
        result = subprocess.run('textline_detect/linedetect', input=input_binary, stdout=subprocess.PIPE).stdout
        detected_boxes = []
        p = 0
        max_block = 0
        count = int.from_bytes(result[p:p+4], byteorder='little')
        p += 4
        for i in range(count):
            id = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            block = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            max_block = max(max_block, block)
            p += 4
            idx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            subidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            subtype = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            pageidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            sectionidx = int.from_bytes(result[p:p+4], byteorder='little', signed=True)
            p += 4
            detected_boxes.append((id,block,idx,subidx,subtype,pageidx,sectionidx))

        features = []
        feature_idx = []
        prev_block = 0
        prev_idx = 0
        vertical = 0
        for id, block, idx, subidx, subtype, pageidx, sectionidx in detected_boxes:
            if id < 0:
                continue

            # 1 vertical
            # 2 ruby (base)
            # 3 ruby (text)
            # 4 space
            # 5 emphasis
            # 6 newline

            ruby = 0
            rubybase = 0
            space = 0
            emphasis = 0

            if prev_block != block:
                prev_block = block
                g = np.zeros([encoder_dim], np.float32)
                g[feature_dim+0] = 5 * vertical
                g[-1] = 5
                features.append(g)
                feature_idx.append((-1,-1,-1,-1,-1))
                prev_idx = -1
            if prev_idx != idx:
                prev_idx = idx
                g = np.zeros([encoder_dim], np.float32)
                g[feature_dim+0] = 5 * vertical
                g[-1] = 5
                features.append(g)
                feature_idx.append((-1,-1,-1,-1,-1))

            if subtype & 2+4 == 2+4:
                ruby = 1
            elif subtype & 2+4 == 2:
                rubybase = 1
            
            if subtype & 8 == 8:
                space = 1

            if subtype & 16 == 16:
                emphasis = 1

            if subtype & 1 == 0:
                vertical = 0
            else:
                vertical = 1
            
            g = np.concatenate([glyphfeatures[id,:], 5*np.array([vertical,rubybase,ruby,space,emphasis,0], np.float32)])
            features.append(g)
            feature_idx.append((id, block, idx, subidx, subtype))

        features = np.array(features, np.float32)
        SP_token = np.zeros([encoder_dim], dtype=np.float32)
        SP_token[0:feature_dim:2] = 5
        SP_token[1:feature_dim:2] = -5

        outdict = {
            'box': [],
            'line': [],
            'block': [],
        }

        cur_i = 0
        prev_j = 0
        result_txt = ''
        keep_back = 0
        linebuf = []
        while cur_i < features.shape[0]:
            r = 0
            s = 0
            for k in range(cur_i, min(cur_i + max_encoderlen - 3, features.shape[0])):
                # space
                if features[k,-3] > 0:
                    r += 1
                # rubybase
                if s == 0 and features[k,-5] > 0:
                    r += 3
                    s = 1
                # ruby
                elif s == 1 and features[k,-4] > 0:
                    s = 2
                elif s == 2 and features[k,-4] == 0:
                    s = 0
            cur_j = min(features.shape[0], cur_i + (max_encoderlen - 3 - r))
            # horizontal / vertical change point
            for j in range(cur_i+1, cur_j):
                if features[j,-6] != features[cur_i,-6]:
                    cur_j = j
                    break
            # double newline (new block)
            if cur_j < features.shape[0]-1 and cur_i+1 < cur_j-1:
                for j in range(cur_i+1, cur_j-1):
                    if features[j,-1] > 0 and features[j+1,-1] > 0:
                        cur_j = j+2
                        break
            # ruby/rubybase separation check
            if cur_j < features.shape[0]:
                # last char is not newline
                if cur_j > 1 and features[cur_j-1, -1] == 0:
                    for j in reversed(range(cur_i+1, cur_j)):
                        # ruby, ruby base
                        if features[j,-4] == 0 and features[j,-5] == 0:
                            cur_j = j+1
                            break

            if prev_j == cur_j:
                keep_back = 0
                cur_i = cur_j
                continue
            print(prev_j,cur_i,cur_j,'/',features.shape[0])
            encoder_input = np.zeros(shape=(1,max_encoderlen, encoder_dim), dtype=np.float32)
            encoder_input[0,0,:] = SP_token
            encoder_input[0,1:1+cur_j-cur_i,:] = features[cur_i:cur_j,:]
            encoder_input[0,1+cur_j-cur_i,:] = -SP_token

            pred = self.call_transformer(encoder_input)
            predstr = ''
            for p in pred:
                if p == decoder_SOT:
                    continue
                if p == decoder_PAD or p == decoder_EOT:
                    break
                if p >= 0xD800 and p <= 0xDFFF:
                    predstr += '\uFFFD'
                elif p < 0x3FFFF:
                    predstr += chr(p)
                else:
                    predstr += '\uFFFD'
            # print(keep_back, predstr)
            result_txt += predstr[keep_back:]
            linebuf += [(prev_j, cur_j, predstr[keep_back:])]

            if cur_j < features.shape[0]:
                k = cur_j - 1
                prev_j = cur_j
                keep_back = 0
                while cur_i < k:
                    # horizontal / vertical change point
                    if features[k,-6] != features[cur_j,-6]:
                        k += 1
                        break
                    # ruby, ruby base
                    if features[k,-5] > 0 or features[k,-4] > 0:
                        k += 1
                        break
                    # newline
                    if k < cur_j - 1 and features[k,-1] > 0:
                        k += 1
                        break
                    # space
                    if features[k,-3] > 0:
                        keep_back += 1
                    if k > cur_j - 3:
                        k -= 1
                    else:
                        break
                if cur_i < k:
                    cur_i = k
                    keep_back += cur_j - k
                else:
                    keep_back = 0
                    cur_i = cur_j
            else:
                break

        line_x1 = -2000
        line_y1 = -2000
        line_x2 = -2000
        line_y2 = -2000
        line_text = ''
        for prev_j, cur_j, predstr in linebuf:
            k_iter = iter(range(prev_j, cur_j))
            try:
                k = next(k_iter)
                for c in predstr:
                    if c in ['\uFFF9','\uFFFA','\uFFFB']:
                        line_text += c
                        continue
                    if feature_idx[k][0] < 0 or c == '\n':
                        if line_text:
                            lineinfo = {
                                'x1': float(line_x1 / resize),
                                'y1': float(line_y1 / resize),
                                'x2': float(line_x2 / resize),
                                'y2': float(line_y2 / resize),
                                'blockidx': blockidx,
                                'lineidx': lineidx,
                                'text': line_text,
                                'aozora': decode_ruby(line_text, outtype='aozora'),
                                'noruby': decode_ruby(line_text, outtype='noruby'),
                            }
                            outdict['line'].append(lineinfo)
                            line_x1 = -2000
                            line_y1 = -2000
                            line_x2 = -2000
                            line_y2 = -2000
                            line_text = ''
                        # print()
                        while feature_idx[k][0] < 0:
                            k = next(k_iter)
                        if c == '\n':
                            continue
                    if c in UNICODE_WHITESPACE_CHARACTERS:
                        line_text += c
                        continue
                    id, blockidx, lineidx, subidx, subtype = feature_idx[k]
                    loc = locations[id]
                    cx = loc[1]
                    cy = loc[2]
                    w = loc[3]
                    h = loc[4]
                    
                    ruby = 0
                    rubybase = 0
                    if subtype & 2+4 == 2+4:
                        ruby = 1
                    elif subtype & 2+4 == 2:
                        rubybase = 1

                    emphasis = 0
                    if subtype & 16 == 16:
                        emphasis = 1
                    
                    if subtype & 1 == 0:
                        vertical = 0
                    else:
                        vertical = 1

                    if ruby == 0:
                        if line_x1 < -1000:
                            line_x1 = cx - w/2
                        else:
                            line_x1 = min(line_x1, cx - w/2)
                        if line_x2 < -1000:
                            line_x2 = cx + w/2
                        else:
                            line_x2 = max(line_x2, cx + w/2)

                        if line_y1 < -1000:
                            line_y1 = cy - h/2
                        else:
                            line_y1 = min(line_y1, cy - h/2)
                        if line_y2 < -1000:
                            line_y2 = cy + h/2
                        else:
                            line_y2 = max(line_y2, cy + h/2)

                    line_text += c

                    boxinfo = {
                        'cx': float(cx / resize),
                        'cy': float(cy / resize),
                        'w': float(w / resize),
                        'h': float(h / resize),
                        'text': c,
                        'blockidx': blockidx,
                        'lineidx': lineidx,
                        'subidx': subidx,
                        'ruby': ruby,
                        'rubybase': rubybase,
                        'emphasis': emphasis,
                        'vertical': vertical,
                    }
                    outdict['box'].append(boxinfo)
                    # print(c, cx, cy, w, h, subtype)
                    k = next(k_iter)
            except StopIteration:
                pass
        else:
            if line_text:
                lineinfo = {
                    'x1': float(line_x1 / resize),
                    'y1': float(line_y1 / resize),
                    'x2': float(line_x2 / resize),
                    'y2': float(line_y2 / resize),
                    'blockidx': blockidx,
                    'lineidx': lineidx,
                    'text': line_text,
                    'aozora': decode_ruby(line_text, outtype='aozora'),
                    'noruby': decode_ruby(line_text, outtype='noruby'),
                }
                outdict['line'].append(lineinfo)

        blockidx = -1
        block_x1 = -2000
        block_y1 = -2000
        block_x2 = -2000
        block_y2 = -2000
        block_text = ''
        for lineinfo in outdict['line']:
            if blockidx != lineinfo['blockidx']:
                if block_text:
                    blockinfo = {
                        'x1': float(block_x1),
                        'y1': float(block_y1),
                        'x2': float(block_x2),
                        'y2': float(block_y2),
                        'blockidx': blockidx,
                        'text': block_text,
                        'aozora': decode_ruby(block_text, outtype='aozora'),
                        'noruby': decode_ruby(block_text, outtype='noruby'),
                    }
                    outdict['block'].append(blockinfo)
                block_x1 = -2000
                block_y1 = -2000
                block_x2 = -2000
                block_y2 = -2000
                block_text = ''
                blockidx = lineinfo['blockidx']
            
            if block_x1 < -1000:
                block_x1 = lineinfo['x1']
            else:
                block_x1 = min(block_x1, lineinfo['x1'])
            if block_x2 < -1000:
                block_x2 = lineinfo['x2']
            else:
                block_x2 = max(block_x2, lineinfo['x2'])

            if block_y1 < -1000:
                block_y1 = lineinfo['y1']
            else:
                block_y1 = min(block_y1, lineinfo['y1'])
            if block_y2 < -1000:
                block_y2 = lineinfo['y2']
            else:
                block_y2 = max(block_y2, lineinfo['y2'])

            block_text += lineinfo['text'] + '\n'
        else:
            if block_text:
                blockinfo = {
                    'x1': float(block_x1),
                    'y1': float(block_y1),
                    'x2': float(block_x2),
                    'y2': float(block_y2),
                    'blockidx': blockidx,
                    'text': block_text,
                    'aozora': decode_ruby(block_text, outtype='aozora'),
                    'noruby': decode_ruby(block_text, outtype='noruby'),
                }
                outdict['block'].append(blockinfo)

        outdict['text'] = result_txt
        outdict['aozora'] = decode_ruby(result_txt, outtype='aozora')
        outdict['noruby'] = decode_ruby(result_txt, outtype='noruby')

        print("---------------------")
        print(decode_ruby(result_txt))

        with open(target_file+'.json', 'w', encoding='utf-8') as file:
            json.dump(outdict, file, indent=2, ensure_ascii=False)


    def run_detector(self, ds, org_img):
        print(org_img.shape)
        print("test")

        locations = [np.zeros(5+4)]
        glyphfeatures = [np.zeros(feature_dim, dtype=np.float32)]
        keymap_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
        lines_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
        seps_all = np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32)
        code_all = []
        for _ in range(4):
            code_all.append(np.zeros([org_img.shape[0] // scale, org_img.shape[1] // scale], dtype=np.float32))

        for n, inputs in enumerate(ds):
            print(n, '/', len(ds))
            x_i = inputs['offsetx']
            y_i = inputs['offsety']
            x_is = x_i // scale
            y_is = y_i // scale
            x_s = width // scale
            y_s = height // scale

            heatmap, features = self.call_detector(inputs['input'])

            mask = np.zeros([y_s, x_s], dtype=bool)
            x_min = int(x_s * (1-self.step_ratio)/2) if x_i > 0 else 0
            x_max = int(x_s * (1-(1-self.step_ratio)/2)) + 1 if x_i + width < org_img.shape[1] else x_s
            y_min = int(y_s * (1-self.step_ratio)/2) if y_i > 0 else 0
            y_max = int(y_s * (1-(1-self.step_ratio)/2)) + 1 if y_i + height < org_img.shape[0] else y_s
            mask[y_min:y_max, x_min:x_max] = True

            keymap_p = sigmoid(heatmap[0,0,:,:])
            line_p = sigmoid(heatmap[0,4,:,:])
            seps_p = sigmoid(heatmap[0,5,:,:])
            code_p = []
            for k in range(4):
                code_p.append(sigmoid(heatmap[0,6+k,:,:]))

            keymap_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(keymap_p * mask, keymap_all[y_is:y_is+y_s,x_is:x_is+x_s])
            lines_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(line_p * mask, lines_all[y_is:y_is+y_s,x_is:x_is+x_s])
            seps_all[y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(seps_p * mask, seps_all[y_is:y_is+y_s,x_is:x_is+x_s])
            for k in range(4):
                code_all[k][y_is:y_is+y_s,x_is:x_is+x_s] = np.maximum(code_p[k] * mask, code_all[k][y_is:y_is+y_s,x_is:x_is+x_s])

            peak = sigmoid(heatmap[0,1,:,:]) * mask
            idxy, idxx  = np.unravel_index(np.argsort(-peak.ravel()), peak.shape)

            for y, x in zip(idxy, idxx):
                if peak[y,x] < self.cut_off:
                    break
                w = np.exp(heatmap[0,2,y,x] - 3) * 1024
                h = np.exp(heatmap[0,3,y,x] - 3) * 1024
                if w <= 0 or h <= 0:
                    continue
                if w > org_img.shape[1] or h > org_img.shape[0]:
                    continue
                ix = x * scale + x_i
                iy = y * scale + y_i

                codes = []
                for k in range(4):
                    codes.append(code_p[k][y,x])

                locations.append(np.array([peak[y,x], ix, iy, w, h, *codes]))
                glyphfeatures.append(features[0,:,y,x])

        locations = np.array(locations)
        glyphfeatures = np.array(glyphfeatures)

        hists = []
        for i in range(locations.shape[0]):
            p = locations[i,0]
            if p < self.cut_off:
                continue
            cx = locations[i,1]
            cy = locations[i,2]
            w = locations[i,3]
            h = locations[i,4]
            x_min = int(cx - w/2) - 1
            x_max = int(cx + w/2) + 2
            y_min = int(cy - h/2) - 1
            y_max = int(cy + h/2) + 2
            hists.append(self.imageHist(org_img[y_min:y_max,x_min:x_max,:]))
        th_hist = np.median(hists) / 5

        idx = np.argsort(-locations[:,0])
        done_area = np.zeros([0,4])
        selected_idx = []
        for i in idx:
            p = locations[i,0]
            if p < self.cut_off:
                break
            cx = locations[i,1]
            cy = locations[i,2]
            w = locations[i,3]
            h = locations[i,4]
            x_min = max(0, int(cx - w/2))
            x_max = min(org_img.shape[1] - 1, int(cx + w/2) + 1)
            y_min = max(0, int(cy - h/2))
            y_max = min(org_img.shape[0] - 1, int(cy + h/2) + 1)
            if self.imageHist(org_img[y_min:y_max,x_min:x_max,:]) < th_hist:
                continue
            area0_vol = w * h
            fill_map = np.zeros([int(w), int(h)], dtype=bool)
            if done_area.size > 0:
                area1_vol = done_area[:,2] * done_area[:,3]
                inter_xmin = np.maximum(cx - w / 2, done_area[:,0] - done_area[:,2] / 2)
                inter_ymin = np.maximum(cy - h / 2, done_area[:,1] - done_area[:,3] / 2)
                inter_xmax = np.minimum(cx + w / 2, done_area[:,0] + done_area[:,2] / 2)
                inter_ymax = np.minimum(cy + h / 2, done_area[:,1] + done_area[:,3] / 2)
                inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
                inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
                inter_vol = inter_w * inter_h
                union_vol = area0_vol + area1_vol - inter_vol
                iou = np.where(union_vol > 0., inter_vol / union_vol, 0.)
                if iou.max() > 0.5:
                    continue
                if inter_vol.max() > area0_vol * 0.75:
                    continue
                idx_overlap = np.where(iou > 0)[0]
                for j in idx_overlap:
                    cx1 = done_area[j,0]
                    cy1 = done_area[j,1]
                    w1 = done_area[j,2]
                    h1 = done_area[j,3]
                    p1x = int(max(cx1 - w1/2, cx - w/2) - (cx - w/2))
                    p2x = int(min(cx1 + w1/2, cx + w/2) - (cx - w/2))+1
                    p1y = int(max(cy1 - h1/2, cy - h/2) - (cy - h/2))
                    p2y = int(min(cy1 + h1/2, cy + h/2) - (cy - h/2))+1
                    fill_map[p1x:p2x,p1y:p2y] = True
                if np.mean(fill_map) > 0.5:
                    continue

            done_area = np.vstack([done_area, np.array([cx, cy, w, h])])
            selected_idx.append(i)

        idx = selected_idx
        selected_idx = []
        for i in idx:
            cx = locations[i,1]
            cy = locations[i,2]
            x = int(cx / scale)
            y = int(cy / scale)
            if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
                if seps_all[y,x] > 0.5:
                    continue
            selected_idx.append(i)

        if len(selected_idx) > 0:
            selected_idx = np.array(selected_idx)

            locations = locations[selected_idx,:]
            glyphfeatures = glyphfeatures[selected_idx,:]
        else:
            locations = np.zeros([0,5+4])
            glyphfeatures = np.zeros([0,feature_dim], dtype=np.float32)

        for i in range(locations.shape[0]):
            cx = locations[i,1]
            cy = locations[i,2]
            w = locations[i,3]
            h = locations[i,4]
            x = int(cx / scale)
            y = int(cy / scale)
            x_min = int(cx / scale - 1)
            y_min = int(cy / scale - 1)
            x_max = int(cx / scale + 1) + 1
            y_max = int(cy / scale + 1) + 1
            if x >= 0 and x < org_img.shape[1] // scale and y >= 0 and y < org_img.shape[0] // scale:
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(org_img.shape[1] // scale, x_max)
                y_max = min(org_img.shape[0] // scale, y_max)
                for k in range(4):
                    locations[i,5+k] = max(np.max(code_all[k][y_min:y_max,x_min:x_max]), locations[i,5+k])

        return locations.astype(np.float32), glyphfeatures, lines_all, seps_all

    @staticmethod
    def imageHist(im):
        def cluster_dist(hist):
            sum_y = np.sum(hist)
            if sum_y == 0:
                return 0

            i = np.arange(hist.shape[0])
            v = hist * i
            mean_y = np.sum(v) / sum_y
            s1 = np.sum(hist[:int(mean_y+0.5)])
            s2 = np.sum(hist[int(mean_y+0.5):])
            if s1 == 0:
                return 0
            if s2 == 0:
                return 0
            g1 = v[:int(mean_y+0.5)]
            g2 = v[int(mean_y+0.5):]
            k1 = np.sum(g1) / s1
            k2 = np.sum(g2) / s2
            dist1 = 256.0
            dist2 = abs(k1 - k2)
            while dist1 != dist2:
                dist1 = dist2
                s1 = np.sum(hist[np.abs(i - k1) < np.abs(i - k2)])
                s2 = np.sum(hist[np.abs(i - k1) >= np.abs(i - k2)])
                if s1 == 0:
                    return 0
                if s2 == 0:
                    return 0
                g1 = v[np.abs(i - k1) < np.abs(i - k2)]
                g2 = v[np.abs(i - k1) >= np.abs(i - k2)]
                k1 = np.sum(g1) / s1
                k2 = np.sum(g2) / s2
                dist2 = abs(k1 - k2)
            return dist1

        maxPeakDiff = -1
        maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,0], bins=256, range=(0,256))[0]))
        maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,1], bins=256, range=(0,256))[0]))
        maxPeakDiff = max(maxPeakDiff, cluster_dist(np.histogram(im[:,:,2], bins=256, range=(0,256))[0]))
        return maxPeakDiff
