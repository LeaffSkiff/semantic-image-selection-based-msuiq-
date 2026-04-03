from collections import OrderedDict

# IMPORTANT NOTES !!!
#   - The score range (min, max) is only rough estimation, the actual score range may vary.

DEFAULT_CONFIGS = OrderedDict(
    {
        'musiq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'koniq10k'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'musiq-ava': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'ava'},
            'metric_mode': 'NR',
            'score_range': '1, 10',
        },
        'musiq-paq2piq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'paq2piq'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'musiq-spaq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'spaq'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
    }
)
