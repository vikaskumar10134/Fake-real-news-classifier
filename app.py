import numpy as np
import pandas as pd
import joblib
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import nltk 

nltk.download('punkt_tab')
nltk.download('stopwords')

# make the stopword set
stop_words = set(stopwords.words('english'))


# preprocess the text
def preprocess_text(text):

    
    # Handle non-string values (NaN, floats, None, etc.)
    if not isinstance(text, str):
        return ""

    # remove unwanted characters (fixed regex too: [] instead of ^…)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # convert to lowercase
    text = text.lower()

    # tokenize
    token = word_tokenize(text)

    # remove stopwords
    token = [word for word in token if word not in stop_words]

    return ' '.join(token)


# load the tfidf vectorizer
tfidf = joblib.load('tfidf.pkl')

# load the model
model = joblib.load('model.pkl')

# add the title
st.title('REAL FAKE NEWS CLASSIFIER')

text_input = st.text_input('Enter the news')

# add the button
if st.button('Predict'):

    # 1. preprocess the text
    
    preprocessed_text = preprocess_text(text_input)

    # 2. vectorizer the text
    vectorizer_input = tfidf.transform([preprocessed_text])

    # 3. predict the news
    result = model.predict(vectorizer_input)

    # check and assign
    if(result == 1):

        st.header('Real news')

    else:
        st.header('Fake news')


# ***********************   real news example    ************************

#***********************   1   ****************************************.

# 1.  The government has announced a new initiative to promote renewable energy sources across the country. This program aims to reduce carbon emissions and create green jobs by investing in solar, wind, and other sustainable energy projects. The initiative includes tax incentives for businesses and homeowners who install renewable energy systems, as well as funding for research and development in clean energy technologies. Officials believe that this move will not only help combat climate change but also boost the economy by fostering innovation and attracting investments in the green sector.

# ***********************  2   ****************************************.

# 'Delhi teen finds place on UN green list, turns glass into sand\n\nNEW DELHI: Udit Singhal , who founded a startup to turn glass bottles into sand , has been named among 17 people in the United Nation’s 2020 Class of Young Leaders for Sustainable Development Goals (SDGs). The programme is a flagship initiative that recognises the efforts of young people in driving action and galvanising others in support of UN’s 2030 agenda for sustainable development.Singhal, an 18-year-old who lives near Mandi House, told TOI, “I feel proud to represent India. Along with 16 others, I’ll work to motivate youths and achieve SDGs. I hope to be able to encourage communities to inculcate better civic sense to create sustainable living spaces.” The teenager attended the UN function online due to the Covid pandemic. “In the normal scenario, I would have gone to New York,” he added.Singhal had founded Glass2Sand in 2019 as a zero-waste system that turns bottles into economically viable sand. “In 2018, I found a pile of bottles at home. During research, I learnt that ragpickers were not interested in collecting glass bottles because of little demand, high transportation cost and storage space. So empty glass bottles often ended up at the landfills. Do you know it takes a million years for a glass bottle to decompose?”Singhal then imported a machine from New Zealand with a special grant from the New Zealand High Commissioner in India and went to work helped by 65 volunteers, six diplomatic missions and four institutions, collecting bottles from across Delhi. “It takes five seconds to turn one empty glass bottle into sand. We have crushed 8,000 kg of bottles into 4,800 kg of high-grade silica sand so far,” he said. “This silica sand is commercially viable because it is used in construction, including roads and as lining in furnaces. It is more valuable than river sand in making concrete.”Singhal studied in The British School in Delhi and is currently a freshman at University College in London where he is a management sciences student.Other young leaders cited by Jayathma Wickramanayake, the UN Secretary-General’s Envoy on Youth, for “leading the way in shaping a more sustainable and inclusive future for all” belong to the United States, Turkey, China, Pakistan, Peru, Egypt, Senegal, Bulgaria, Nigeria, Brazil, Liberia, Ireland, Colombia, Australia, Uganda and Bangladesh.'



# ***********************   fake news example    ************************

# ***********************  1   ****************************************.
#
# 'The Simpsons have been known to come up with episodes that seem to predict the future, including the election of Donald Trump as president.\n\n\n\nBut did The Simpsons creator Matt Groening and his team of writers predict George Floyd\'s death and the protests that followed afterwards?\n\nProtests broke across several cities in the United States following the death of an African-American man, George Floyd, who died after being pinned to the ground by police officers in Minneapolis, Minnesota on May 25, 2020. Footage shot by witnesses showed a white police officer, Derek Chauvin, with his knee on Floyd\'s neck for close to nine minutes as the latter stopped responding.\n\nOn 1 June, 2020 a bunch of images started doing the rounds on social media that made it seem as if the critically acclaimed show had accurately predicted the future again. The most popular image was of The Simpsons character Chief Clancy Wiggum, strangling a black man with his foot while Lisa Simpson has a \'Justice For George\' protest placard. The claim is that it appeared in the series in the 1990s.\n\nThere\'s also an image of what appears to be the police precinct in Minneapolis catching fire shown juxtaposed the Springfield police station catching fire.\n\nThey predicted George Floyd\'s death on the Simpsons...coincidence? pic.twitter.com/oOx4W9HhlJ — Heba A. 🇵🇸🇺🇸 (@heebz101) June 1, 2020\n\nThese images when put together seemed to indicate that animated series had accurately predicted police brutality case like George Floyds\' and the aftermath that followed.\n\nFact Check\n\nA closer look at the image of Chief Wiggum and the black man shows an artist\'s signature at the bottom right side.\n\n\n\n\n\n\n\nBOOM looked for \'Yuri Pomo\' on Twitter and Instagram, and it turns out Pomo is an artist who created this cartoon depicting police brutality on May 30, 2020 to raise awareness about George Floyd\'s death. It is not a scene from The Simpsons.\n\nThere\'s an image of what appears to be the precinct in Minneapolis, Minnesota that was set on fire that\'s juxtaposed with the Springfield police station ablaze. The 3rd Precinct of the Minneapolis police station was set on fire to protest George Floyd\'s death. BOOM could not independently verify if the building in the photo was that of the Precinct.\n\n\n\nThe next image is taken from an episode of The Simpsons. A YouTube search for the keywords \'The Simpsons Springfield Police Station Catching Fire\' led us to find this clip:\n\nThis clip from The Simpsons season 11, episode 06, titled \'Hello Gutter, Hello Fadder.\' The episode shows the Springfield police station on fire but is unrelated to a riot, or even a larger part of the episode. Homer is rushing to work and tries to take a few short cuts, so Chief officer Wiggum stops him and says \'Where\'s the fire?\' to ask him what was the hurry in him driving rashly. Homer says, "Over there." and points to the Springfield police station on fire. There is no background of rioting or protests, and is, in fact, a short joke skit that The Simpsons is known for.'

# ***********************  2   ****************************************.


#'Several users of social media are calling out photos of anti-Citizenship Amendment Act (CAA) protesters, with visible bandages, as imposters with no real injuries. According to them, the photos, which shows the protesters wearing bandages over pieces of clothing like \'hijabs\' and jackets gives away their bluff. However, the photographer of these images, Zafar Abbas, told BOOM that the protesters deliberately wore bandages during anti-CAA protests in New Delhi on December 29, as a mark of solidarity with a student - Mohammed Minhajuddin - who lost an eye during earlier protests on December 15 due to police brutality.\n\nOn December 15, Jamia Millia student Minhajuddin lost his left eye in protests against the CAA, reportedly due to police lathicharge.\n\n\n\nThere are various such photos of the protests taken by Abbas in New Delhi, of which a collage of two can be seen below.\n\n\n\n\n\n\n\n\n\n\n\nBOOM received the photos multiple times on its helpline (7700906111). It has the caption, "They have bandage over their Hijaab and over their jacket... Do you need more proof of how fake these protests are."\n\n\n\n\n\n\n\n\n\n\n\n\n\nUser of Twitter including Paresh Rawal, amplified the claim which calls such bandaged protesters out. His tweet was retweeted more than 2,700 times, and liked more than 13.6 thousand times.\n\n\n\n\n\n\n\n\n\n\n\n\n\nAn archived version of his tweet can be found here.\n\n\n\n\n\n\n\n\n\nMore tweets later emerged of other photos from the protests, showing bandages on their clothing and calling them out.\n\nThe same picture with similar claims is also online, which have been worded differently.\n\nOk let me now teach you all guys running fake propaganda, some basics of #wound #dressing.\n\nFirst you remove the soiled clothes\n\n\n\nThen you clean the wound\n\n\n\nAnd then you apply bandages after debriding the dirty tissue.\n\n\n\nPS: it is never applied over jacket and hijab. #CAA_NRC pic.twitter.com/66ZpqDwfaW — TheSpeakingScalpel (@DrSaurav5) January 1, 2020\n\n\n\n\n\nFactCheck\n\n\n\n\n\n\n\nOn looking through many more such pictures on Twitter, many of these images at first glance carry the watermark of \'Zafar Abbas\', who is a journalist and works with the Millennium Post.\n\n\n\n\n\n\n\n\n\n\n\n\n\nBOOM got in touch with Abbas, asking him the date and nature of the photographs, to which he replied:\n\n\n\n"I took the photo on 29th December ( Sunday ) when the students of Jamia Millia were protesting in an innovative way by tying a bandage around one eye in solidarity with Minhajuddin, a student who lost an eye in Delhi police crackdown."\n\n\n\n\n\n\n\nOn the nature of fake claims that has emanated from his pictures, he told BOOM:\n\n"I was surprised to see trolls and even some verified twitter handles used my pic without checking the background of the symbolic protest and tried to mislead people."\n\nThe original images being circulated on social media, along with many other images from the protest have been tweeted by Abbas through his Twitter account and can be seen below.\n\n\n\n\n\n\n\n\n\nEye bandaged : Jamia students bandage one eye in solidarity with fellow student Minhajuddin who lost an eye in @DelhiPolice crackdown on December 15. Innovative way of protest.#CAA_NRC_NPR #CAAProtest #JamiaWalaBagh #JamiaMilliaIslamia pic.twitter.com/AzRO5roHjR — Zafar Abbas (@zafarabbaszaidi) December 29, 2019\n\n\n\n\n\nSimilar images of the protests on December 29 can also be found through other sources. News18 too covered the protests, and one can see a compilation of pictures from the protests where they have outlined that the bandages are a mark of solidarity.\n\n\n\n\n\n\n\n\n\nThe CAA was signed into law by President Kovind on December 12, 2019 and since then there have been widespread protests across the country against the Act that have caused at least 25 casualties. Opponents of the Act - which grants expedited Indian citizenship to non-Muslim minority refugees fleeing religious persecution from Afghanistan, Pakistan and Bangladesh - state that the Act, in conjunction with the a potential nation-wide National Registry of Citizens (NRC) could directly render sections of Indian Muslim stateless. The government refutes this, stating that while a nation-wide NRC has not even been announced, the CAA is a humane law to provide shelter to persecuted minorities with nowhere to go.\n\n\n\n\n\n\n\n'
