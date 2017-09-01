# 모니터링 패키지 로딩.
from .monitor import Monitor

# 어큐러시 모니터링 클래스
class AccuracyMonitor(Monitor):
    
    # 멤버 변수
    stat_name = 'accuracy'
    
    # 생성자
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', '%')          
        kwargs.setdefault('precision', 2)
        super(AccuracyMonitor, self).__init__(*args, **kwargs)
        
    # 어큐러시 리턴 메써드
    def _get_value(self, iteration, input, target, output, loss):
        batch_size = input.size(0)
        predictions = output.max(1)[1].type_as(target)
        correct = predictions.eq(target)
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.sum()
        return 100. * correct / batch_size
