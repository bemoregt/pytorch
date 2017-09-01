# 모니터링 패키지 로딩.
from .monitor import Monitor

# 어큐러시 모니터링 클래스
class AccuracyMonitor(Monitor):
    
    # 멤버 변수
    stat_name = 'accuracy'
    
    # 생성자
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', '%')          # 단위 퍼센트가 디폴트        
        kwargs.setdefault('precision', 2)       # 프리시젼은 2가 디폴트
        super(AccuracyMonitor, self).__init__(*args, **kwargs)      # ??
        
    # 어큐러시 리턴 메써드
    def _get_value(self, iteration, input, target, output, loss):
        batch_size = input.size(0)                          # 배치 사이즈
        predictions = output.max(1)[1].type_as(target)      # 예측값을 클래스명으로
        correct = predictions.eq(target)                    # 예측값이 타겟값과 같은지
        if not hasattr(correct, 'sum'):                     # ??
            correct = correct.cpu()
        correct = correct.sum()                             # 맞은것 더하기
        return 100. * correct / batch_size                  # 맞은것 평균 퍼센트로 리턴.    
