
# coding: utf-8

# In[24]:

#배열 설명 [gender, age, emotion, returnMessage]
#gender 0 남 , 1 여
#age 0 1살 , 1  5살, 2 10살, 3 16살, 4 28살, 5 51살, 6 75살
#emotion 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
def callFortunesDataSet():
    dataSet = [
        (0,0,0,"""오!!! 난 알수 있어요! 레오나르도 다빈치가 일생에 걸쳐서 사용한것보다 높은 지능의 잠재력을 가지고 있어요!"""),
        (0,0,1,"뭐든 미리 다 알고 있다면 시시하지 않겠어요? 상상할 거리가 없어지잖아요."),
        (0,0,2,"얼굴에서 강한 기운이 느껴지는군, 마치 포효하는 백두산 호랑이처럼… "),
        (0,0,3,"쌩~ 쌩~ 찬바람아 불어라. 추울수록 늠름해지고 마음이 굳세지는 어린이"),
        (0,0,4,"실망하는것보다 아무것도 기대하지 않는게 더 나쁘다고 생각해요"),
        (0,0,5,"다시 봄이 오고 이렇게 숲이 눈부신 것은 파릇파릇 새잎이 눈뜨기 때문이지"),
        (0,0,6,"""작은 바람에도 살아 쓸리는 여린 풀잎 그러나 우람한 나무로 많은 이들의 시원한 그늘이 되어줄거야 """),
        (0,0,7,"흔들어주고, 속삭여주고, 간질여주고, 그런 바람이 없으면 나무가 얼마나 재미없겠니"),
        (0,1,0,"어리지만 든든한 우리의 길잡이가 되어 다오!"),
        (0,1,1,"사자 같은 얼굴은 가진 사나이! 무럭무럭 자라라!"),
        (0,1,2,"엄마 아빠의 꾸지람 그속에 걱정. "),
        (0,1,3,"달려가고, 넘어지고, 올라가고 그런 바람 재울 날 없는내가 있어서 우리 엄마 얼마나 재밌겠니."),
        (0,1,4,"모험을 하세요! 모험에서 얻어지는건 초코렛보다 달콤해요"),
        (0,1,5,"이 세상의 희망. 우리나라의 희망. 우리들의 희망. 기 할테니까 무럭무럭 커야되요!~"),
        (0,1,6,"나는 보인다. 이 아이가 생각이 많다는걸. 슬픈얼굴에  총명함. "),
        (0,1,7,"흠… 뭔가 특별해. 이 얼굴. 가장 유능한 사람은 가장배움에 힘쓰는 사람이야!"),
        (0,2,0,"작은 그릇에는 적은 양에 물이 담길 것이고 큰 그릇에는 그만큼의 물이 담기는 뜻을 알아야 합니다. "),
        (0,2,1,"흉한 것이 변하여 나에게 길함을 얻게 해 줄것이니 어렵고 귀찮은 일이라도 무시하거나 배척하지 말고 자진하여 실행하면 좋은 결과를 보게 될 일진이다."),
        (0,2,2,"겉은 화려하고 속은 비었으니 겉치장에 신경쓰는 일은 버려야 합니다.."),
        (0,2,3,"화창한 봄날에 꽃과 나비를 보는 것과도 같은 날이니 만사가 마음먹은대로 될것입니다."),
        (0,2,4,"뛰어난 지혜와 지모가 탁월하더라도 시대를 잘못 타고나면 하찮은 건달에 지나지않을 수도 있습니다."),
        (0,2,5,"하나가 둘이 되고 둘이 셋이되니 날로 번창하는 좋은 운기가 있을 것입니다."),
        (0,2,6,"나그네가 어두운 산속에서 길을 잃은 것과도 같으니 도모하는 일이 있다면 타의 조력도 받지 못하고 어려움만 중첩하게 되는 불길한 날입니다."),
        (0,2,7,"작은 그릇에는 적은 양에 물이 담길 것이고 큰 그릇에는 그만큼의 물이 담기는 뜻을 알아야 합니다. "),
        (0,3,0,"모든일이든 처음과 끝을 같게 할 것이니 중도에 좌절하게 되면 시작하지 않음만도 못하게 될 것입니다. """),
        (0,3,1,"한차례 비가 온 하늘이 맑게 개이니 무지개가 아름답듯이 힘든일이 지나가면 좋은일이 올것입니다."),
        (0,3,2,"뛰어난 지혜와 지모가 탁월하더라도 시대를 잘못 타고나면 하찮은 건달에 지나지 않음과도 같은 날입니다."),
        (0,3,3,"혼자하는 일은 없습니다. 남에게 도움을 요청하거나 자문을 구하는 일을 자존심 상하거나 창피스러운 일로 생각하면 안됩니다."),
        (0,3,4,"무리하여 진행하는 일을 하기에는 좋지 않습니다. 잠시 쉬어가는것도 방법중 하나입니다."),
        (0,3,5,"흉함이 있은 후에 길함이 찾아올것입니다. 처음 곤란한 일을 당할지라도 좌절하지 말고 꾸준히 밀고 나간다면 결말이 좋을것입니다."),
        (0,3,6,"주위의 도움을 받아 이룸이 많을 것입니다. 주위에 인덕이 많아서 도움을 받는것도 보이지 않는 하나의 재산입니다.  "),
        (0,3,7,"모든일이든 처음과 끝을 같게 할 것이니 중도에 좌절하게 되면 시작하지 않음만도 못하게 될 것입니다. "),
        (0, 4, 0, "'삶이 그대를 속일지라도 슬퍼하거나 노여워말고' 그것에 익숙해지라. "),
        (0, 4, 1, "진정으로 웃으려면 고통을 참아야하며 , 나아가 고통을 즐길 줄 알아야 해 - 찰리 채플린"),
        (0, 4, 2, "자신감 있는 표정을 지으면 자신감이 생긴다 -찰스다윈"),
        (0, 4, 3,"피할수 없으면 즐겨라 - 로버트 엘리엇"),
        (0, 4, 4, "행복한 삶을 살기위해 필요한 것은 거의 없다.-마르쿠스 아우렐리우스 안토니우스"),
        (0, 4, 5, "오랫동안 꿈을 그리는 사람은 마침내 그 꿈을 닮아 간다, -앙드레 말로"),
        (0, 4, 6, "스스로에게 늘 친절하고 자아 존중감을 잃지마라."),
        (0, 4, 7, "1퍼센트의 가능성, 그것이 나의 길이다. -나폴레옹"),
        (0, 5, 0, "인간은 얼굴을 붉히는 유일한 동물이다. 또한 그렇게 할 필요가 있는 동물이다-마크 트웨인"),
        (0, 5, 1, "어떻게 늙어야 하는지를 알고 있는 사람은 드물다-라 로슈푸코 "),
        (0, 5, 2, "사십세가 지난 인간은 자신의 얼굴에 책임을 져야 한다-링컨"),
        (0, 5, 3, "상황은 비관적으로 생각할 때에만 비관적으로 된다-발리 브란드"),
        (0, 5, 4, "행복이란 타인을 행복하게 해주려는 노력의 부산물이다. -파머"),
        (0, 5, 5, "인간의 나이는 가장 중요한 일이다. 그러나 여자들은 이것을 인정치 않고, 남자들은 이것에 대한 행동을 하지 않는다-레오날드 쉬프"),
        (0, 5, 6, "아름다움을 발견하고 즐겨라 약간의 심미적 추구를 게을리 하지마라. 그림과 음악을 사랑하고 책을 즐기고 자연의 아름다움을 만끽하는 것이 좋다 "),
        (0, 5, 7, "무언가 큰 일을 성취하려고 한다면, 나이를 먹어도 청년이 되지 않으면 안 된다-괴테 "),
        (0,6,0,"새 빗자루가 조금 더 먼지를 잘 쓸지 몰라도 오래된 빗 어디가 어딘지 알죠"),
        (0,6,1,"사람의 냄새와 사람의 껍질을 벗고서도 또 사람. 그게  우리네요. 아 참… 저는 컴퓨터 입니다…"),
        (0,6,2,"거칠은 바람 속에 등에 진실한 땀을 흐른 흔적이 보여요"),
        (0,6,3,"""나이는 별로 중요하지 않아요. 만약 당신이 우유이지 않는 이상."""),
        (0,6,4,"나이 들었다는건 지금 당신의 나이보다 15년 더 많았을 얘기 입니다."),
        (0,6,5,"아프더라도 그 아픔을 즐기자. 행복은 긍정에서 시작되고감사와 함께 자라고, 사랑으로 완성된다. "),
        (0,6,6,"눈길을 걸어 갈 때 어지럽게 걷지 말기를, 오늘 내가  길이 훗날 다른 사람의 이정표가 되리니... - 백범 김구"),
        (0,6,7,"소중한 순간이 오면 따지지 말고 누릴것. 우리에게 내일이 있으리란 보장은 없으니까"),
        (1,0,0,"완전 천사인줄 알았어요! 우리 같이 걸으면 하얀 구름  걷는것 같아요! 아… 컴퓨터가 걸을수 있나? "),
        (1,0,1,"이제 부터 발견할 일이 잔뜩 있다는건 멋진 일이니까요"),
        (1,0,2,"넌 누구도 갖지 못한 별을 갖게 될꺼야!"),
        (1,0,3,"세상대로 되지 않는다는건 정말 멋진것 같아요. 생각지도못 했던 일이 일어난다는 거니까요!"),
        (1,0,4,"헉… 나는 컴퓨터인데 내 매마른 영혼을 따뜻하게 해주네 ㅠㅠ"),
        (1,0,5,"정말로 행복한 나날이란 멋지고 놀라운 일이 일어나는 날 아니라 진주 알들이 하나하나 한 줄로 꿰어지듯 소박 자잘한 기쁨들이 조용히 이어지는 날들인것 같아요."),
        (1,0,6,"미세한 슬픔에도 상처받아 우는 작은 별빛. 그렇지만 작 별빛도 이 세상을 밝혀주기 때문에 아름다워요~"),
        (1,0,7,"놀란 얼굴이 마치 사슴 같아요. 사슴처럼 아름다워질것이 나중에 커서 세상에 좋은 소식을 전해줄거예요!"),
        (1,1,0,"자유롭고 상상력을 지닌 어린이가 바로 제 앞에 있어요!"),
        (1,1,1,"그 사랑스런 눈빛. 아름다움이 또한 어디에 있으랴"),
        (1,1,2,"세상에 쓸모 없는 일은 없어요. 많이 보고 느끼세요!~"),
        (1,1,3,"사막이 아름다운 것은 어딘가에 샘을 감추고 있기 때문이야!"),
        (1,1,4,"처음부터 자신있는 사람은 없어요. 힘내요 우리 친구!~"),
        (1,1,5,"저렇게 산이 듬직한 것은 쑥쑥 새싹들이 키 크기 때문이지"),
        (1,1,6,"슬퍼하지 마라요. 얼굴에는 밤하늘의 별처럼 빛나는 은하수 같이 무궁무진한 미래가 펼쳐질거라고 쓰여져 있어요"),
        (1,1,7,"아침은 어떤 아침이든 즐겁죠. 오늘은 무슨 일이 일어날지 생각하고 기대하는 상상의 여지가 충분히 있거든요."),
        (1,2,0, "오늘은 모든 일에 성의를 갖고 신의를 도모해 나가면 만사가 어려움 없이 풀려나가게 될 것입니다. 아무쪼록 매사에 성심 성의껏 임하여 보람된 하루가 되기를 빌어 마지않겠습니다."),
        (1,2,1,"그 동안 해결될 기미가 보이지 않던 일도 차츰 풀려 나가게 될 것입니다. 앞으로 하루가 다르게 번창해 나가게 될 것입니다"),
        (1,2,2,"오늘은 입 조심 말조심을 해야 합니다. 생각 없이 내뱉은 말 한마디가 큰 불화를 불러일으킬 수도 있다는 사실을 명심하시고 말을 가려서 하도록 하 십시오."),
        (1,2,3,"앞으로 나아 갈수록 좋은 날 입니다. 중도에서 단념해 버리고 마는 것은 아까운 기회를 스스로 차버리고 마는 결과를 초래하게 될 수 있으므로 포기하 지 말고 꾸준히 밀고 나가시기 바랍니다."),
        (1,2,4,"오늘은 물고기와 용이 모두 바다로 모이는 형국입니다. 선배의 도움으로 뜻밖의 성공을 하게 됩니다. 또한 여러 사람이 합심하여 협동하는 일이라면 대성할 수 있겠습니다."),
        (1,2,5,"강한 것이 부드러운 것을 기꺼이 따르는 형국입니다.운이 좋지 않기는 하지만 결코 악운은 아닙니다. 친구를 찾아 가거나 선배를 만 나게 되면 돌파구가 마련되고 새로운 도약을 기대할 수 있습니다."),
        (1,2,6,"온유하고 겸손하고 유순하게 행동하면서 대지가 하늘을 받들 듯이 사랑과 용서를 베풀면 온갖 복을 누리며 만사형통 할 것입니다."),
        (1,2,7,"여섯마리의 용을 타고 하늘에 오르는 형국입니다. 만족할 만큼 매우 좋 은 날입니다. 망설이지 말고 지금하고 있는 일을 그대로 밀고 나가는 것이 바 람직합니다."),
        (1,3,0,"가난이 풍요로 슬픔이 기쁨으로 돌아서는 형국입니다.헤어졌던 가족이 한 자리에 모이고 그 동안에 손해가 이익되어 돌아오는 날 입니다."),
        (1,3,1,"앞으로 나아 갈수록 좋은 날 입니다. 중도에서 단념해 버리고 마는 것은 아까운 기회를 스스로 차버리고 마는 결과를 초래하게 될 수 있으므로 포기하 지 말고 꾸준히 밀고 나가시기 바랍니다."),
        (1,3,2,"여섯마리의 용을 타고 하늘에 오르는 형국입니다. 만족할 만큼 매우 좋 은 날입니다. 망설이지 말고 지금하고 있는 일을 그대로 밀고 나가는 것이 바 람직합니다."),
        (1,3,3,"원하는 바가 성취 될 수 있겠습니다. 특히 재능을 인정 받게될 좋은 기 회이므로 학생은 논문을 제출하고 직장인은 아이디어를 제출하도록 하십시오."),
        (1,3,4,"천리 길도 한 걸음부터이니 계단을 하나씩 밟아 올라가듯 차분하면서도 침 착하게 행동하시기 바랍니다. 그리하면 반드시 좋은 일이 있게 될 것입니다."),
        (1,3,5,"노력한 만큼 돌아오고 남에게 자기 능력을 인정받을 수 있는 날입니다. 바른 마음과 굳은 신념을 가지고 용기 백배하여 뛰어보시기 바랍니다. "),
        (1,3,6,"비교적 원만한 편이라고 할 수 있으나 당신이 지나치게 자존심 만을 내세운다면 좋았던 관계가 나쁜 쪽으로 발전 할 수 있음을 명심해야 합니다."),
        (1,3,7,"온유하고 겸손하고 유순하게 행동하면서 대지가 하늘을 받들 듯이 사랑과 용서를 베풀면 온갖 복을 누리며 만사형통 할 것입니다."),
        (1, 4, 0, "웃자! 웃어라! 남자든 여자든 눈이 마주치면 웃어라. 꼭 호탕한 웃음이 아니더라도 미소만으로 충분하다. "),
        (1, 4, 1, "행복은 습관이다,그것을 몸에 지니라 -허버드"),
        (1, 4, 2, "배려해줘라. '내가 먼저' 라는 생각을 버릴 것. 배려심 많은 사람을 싫어하는 사람은 없다."),
        (1, 4, 3, "모든 것은 지금 구해야 한다. 지금 당신, 그 자체가 하나의 빛이다."),
        (1, 4, 4, "사랑과 친절을 구분하자.　모든 사람에게 친절한 것까지는 좋으나 어떤 사람에게는 상처가 될 수 있다. 우유부단한 행동은 하지 말 것. "),
        (1, 4, 5, "긍정적인 사람으로 보여라.　매사에 부정적인 사람의 인상이 좋을 리 없다. 희망차고 기분 좋은 사람으로 보이려면 먼저 생각을 바꿀 것."),
        (1, 4, 6, "눈물 남발 금지!　눈물을 자주 흘리지 말아라. 위로해줄 지는 모르지만 당신의 매력지수는 그만큼 내려가고 있을 것."),
        (1, 4, 7, "솔직하게! 하지만 신비롭게! 솔직하게 자신을 보여 주는 것은 좋지만 어느정도의 신비감은 그 사람에 대한 궁금증으로 남겨두는 것이 좋다."),
        (1, 5, 0, "명성은 얻는 것이요. 인격은 주는 것이다. - 테일러"),
        (1, 5, 1, "여성은 가장 여성다울 때에 완전하다 - 윌리엄 글래드스턴"),
        (1, 5, 2, "적을 사랑하라. 그들은 너의 결점을 말해주기 때문이다-벤지만 프랭클린"),
        (1, 5, 3, "평온한 바다는 결코 유능한 뱃사람을 만들 수 없다."),
        (1, 5, 4, "위트가 있는 여자는 보석과 같다. 위트가 있는 아름다움은 곧 힘이다 - 조지 메레디스"),
        (1, 5, 5, "훌륭한 인간이 되기 위해서는 나이를 먹는 것이 필요하다. 나는 실수를 범하려 할 때마다 그것은 전에 범했던 실수란 것을 깨닫게 된다-괴테"),
        (1, 5, 6, "사람은 누구나 마음의 집을 마련 하지만, 나중에는 그 집이 마음을 가두어 버린다-에머슨"),
        (1, 5, 7, "명성은 얻는 것이요. 인격은 주는 것이다. - 테일러"),
        (1,6,0,"당신의 눈을 보고 있으면 맑은 호수를 보는것 같아요 "),
        (1,6,1,"미모는 눈을 매료시키지만, 상냥한 태도는 영혼을 매료시킨다. -나레"),
        (1,6,2,"사막이 아름다운 것은 어딘가에 샘이 숨겨져 있기 때문이죠"),
        (1,6,3,"그대가 배운 것을 돌려줘라. 경험을 나눠라. -도교"),
        (1,6,4,"생각해보세요. 문득 뒤얽힌 날들 속에 그 옛날 어린시절의 마음으로 돌아가 바라보면 다시 환한 또 하나의 행복이 당신을 바라보고 있어요"),
        (1,6,5,"선한 눈빛을 가지고 있으세요. 여러 세월동안 좋은 것들을 많이 보셨나봐요"),
        (1,6,6,"사람들은 슬픈표정을 약해서 짓지 않는거 같애요. 너무 오랫동안 강하게 지내와서 그런거죠. "),
        (1,6,7,"내 마음 뛰노라, 나 어려서 그러하였고, 지금도 그러할지다")
    ]
    
    return dataSet
array = callFortunesDataSet()


# In[26]:

def  makeFortunesByPhysiognomy(gender, age, emotion) :
    msg = ""
    
    array = callFortunesDataSet();
        
    for (arr_gender, arr_age, arr_emotion, arr_msg) in array :
        if arr_gender != gender:
            continue
        
        if arr_age != age:
            continue
        
        if arr_emotion != emotion:
            continue
        msg += arr_msg
        
    return msg

#print makeFortunesByPhysiognomy(1,1,0)
#print makeFortunesByPhysiognomy(1,6,7)

