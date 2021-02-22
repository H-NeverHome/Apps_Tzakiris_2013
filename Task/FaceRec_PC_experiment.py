from psychopy import core, visual, event, gui, logging
import os, random, glob, csv, itertools, collections
from random import randint
import pandas as pd
import pickle


'''
This experiment is inspired and conceptualy replicates the experimental paradigm from Apps & Tsakiris; Nature Comm., 2013:'Predictive codes of familiarity and context during the perceptual learning of facial identities'
The experiment was designed to measure facial identity recognition in a context and familiarity dependent manner.
In this experiment, there are 24 identities with with possible presentations from 3 viewpoints (front, left, right).
These identities are presented in a pseudorandomized way where only the number of presented faces was control. In order to reproduce the original experiment we had to reverse engenieer the trial structure from one of the figures and limited information on the supplementary material in Apps and Tzakiris original article, we reproduced the paradigm to the best of our ability. If you want to run this experiment as close as possible to the original you:
-------------------------------------------------------------------------------
!!! MUST NOT EVER CHANGE THE TRIAL ORDER!!! <3
-------------------------------------------------------------------------------
For more information on this process please contact HAUKE NIEHAUS or Porf. Dr. DOMINIK ENDRES 
-------------------------------------------------------------------------------
In this experiment there are 189 presentations 15 identities are presented 4 times from each side.
The remaining 9 are thrown into the mix just once, these are presented by to the end, and mostly for reliability.

We are sure that there are potentially loads of things that could be improved in this script but it is written with love, so please, please be gentle, :D
A good way to get familiar with the code (if you are new to python) is by starting to read from the highlighted point at the end.
(If you aren't new it might still help to understand what how the script is structured :D)
'''

#empty list for key responses
responses=[]

#dictionary of already shown pictures with counter
alreadyshown = {}

#list where the pictures are loaded initially
facesfront = []
facesleft = []
facesright = []

#pool list with all stimuli
facepool = []

matrix = [facesfront, facesleft, facesright]

# Stimuli order was derived only from information in the original article (Apps & Tzakiris; Nat comm, 2013). 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!never touch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#order = [14,8,8,13,8,8,8,10,10,13,10,15,14,12,15,8,12,14,12,13,13,8,8,13,10,15,10,14,14,13,12,12,2,15,8,14,10,8,10,10,8,12,8,10,1,10,15,2,13,2,2,12,13,4,11,14,6,15,13,2,15,11,4,12,10,14,7,10,7,14,1,15,11,1,11,13,4,6,11,1,14,14,15,7,12,1,14,5,13,12,2,2,15,15,15,13,2,2,12,12,6,5,5,3,4,9,4,11,3,4,6,22,7,11,5,4,1,3,7,1,4,6,9,3,2,6,1,4,9,4,11,1,4,5,4,2,5,7,3,1,3,5,1,3,2,9,3,1,6,5,18,7,6,20,5,21,23,9,16,6,24,11,19,7,9,7,6,17,7,6,5,11,5,11,3,6,7,11,9,9,7,5,9,9,9,9,3,3,3]
order = [14,8,8,13,8,8,8,10,10,13,10,15,14,12] # uncomment shortened order for debugging or short trainning session

#jitter_array = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6] * 27 #complete array to ensure we are sampling the BOLD at different times
jitter_array = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6] * 2 #debuger/training array 

faceindex = [i for i in range(0,24)]
random.shuffle(faceindex)

print(faceindex)

frontcounter = {}
leftcounter = {}
rightcounter = {}

# Example for logging
logging_list = []
log_dict = {"subjectID": "", "onset": [], "event": []}

onset_dict = []
event_dict = []

#randomizes ratingscale between participants
rating = random.randint(1,2)

#Almost through the first bit of the experiment, so exciting. :D
def ParticipantInfo():

	#Interface asking for the participant ID. Literally the little grey box that pops up. GUI has to be imported from psychopy so whenever something doesn't work check if all required modules are imported first.
    partinfo = gui.Dlg()

	#Literally the question the little interface is supposed to ask, there can be a lot of things on there. Have a look a the GUI documentation of psychopy.
    partinfo.addField("Subject ID:")
	
	#You have to tell psychopy to show this otherwise it will only be created, this function() is imported with GUI.
    partinfo.show()
	
	#Responses=[] is a list that will pop up multiple times. It's like a little working memory, the participant info as well as recent button presses and the names of the images are saved here. This is the basis for checking if the participant responded correctly.
    responses.append(partinfo.data)
    log_dict['subjectID'] = partinfo.data[0]

                          
						  
	#That's it. This was the last function() in StartExperiment(). So we have to go back to our first Block at the bottom to see which function is called next. 
	#It should be the Welcomescreen.
	#I implemented this a long time ago and the whole glob.glob business I can no longer explain but if you google how to load image files into psychopy lists = [] you will find a lot of way of doing this. I found this way to be the most elegant and efficient. If you know a better one, use that instead. ;)
		
def AddStimuli():
	#draw faces from folder	
    front = [os.path.join('\\Users\\path\\to\\stimuli\\faces\\front', image) for image in glob.glob( 'C:\\Users\\path\\to\\stimuli\\faces\\front\\*.bmp')]
	
	#I just loaded the file into a list = [] but this list is contained within a function so no other function() can easily access this list (there are ways but that makes following steps more complicated and I needed it to be simple :D).
    for pic in front:
        facesfront.append(pic)  #The list=[] facesfront=[] can be found up top. We append() (this is a list function() and standard in python) every picture in our list here to the list up top. Return might be a way of making the list=[] faces=[] accessible from outside the function().
    left = [os.path.join('\\Users\\path\\to\\stimuli\\faces\\left', image) for image in glob.glob('\\Users\\Nestor\\Desktop\\FRAPPS\\ExperimentNstimulus\\faces\G3\\left\\*.bmp')]


	#The same we did for the faces from the front we now do for the faces from the left and the right angle.
    for pic in left:
        facesleft.append(pic)
    right = [os.path.join('\\Users\\path\\to\\stimuli\\faces\\right\\', image) for image in glob.glob('\\Users\\path\\to\\stimuli\\faces\\right\\*.bmp')]

    for pic in right:
        facesright.append(pic)
		
                          #Make sure to have 3 folders one for each viewpont from which you can draw your files, otherwise you will need to change stuff. :)
                          #Now back to where we came from: StartExperiment(), the next function that is being called is poolcreator() I will see you
						  #there.
						
#This function takes an image from our stimui list=[] and presents it, and asks afterwards if the person has seen that before.
def PresentStimuli(in_img): #include the counter (in_img, in_counter)
def PresentStimuli(in_img,in_counter): 
#Here we save the name of the image in our "memory" we called it responses=[] and saved the participant ID in it. Atm an image is called f03r for female number 3 viewed from the right. [0,3] shortens it to f03 which will later be used for a viewindependent check of reoccurence.
    responses.append(in_img.name[0:3])
	
	#Prepare Image
    in_img.draw()
    logging_list.append(str(clock.getTime()) + str(in_img.image))
	
	#Here we log something in our logfie the first time, namely the name of our image.
    logging.log(level=28, msg= in_img.name)
    onset_dict.append(str(clock.getTime()))
    event_dict.append(in_img.name)
	
	
	#Here the image is actually being presented.
    win.flip()
	
    #Makes sure the image is presented exactly 750ms.
    core.wait(0.75)
	
    question = visual.TextStim(win, text = 'Have you seen this person before?',
    pos = (0,0.5), height=0.1, wrapWidth=None, ori=0, depth=0.0)#Creates a Textstimulus like earlier.

    if rating == 1:
	
		#Indicator, there are two to randomly present yes and no on the screen.
        #ratingScale = visual.RatingScale(win, scale= None, choices=['yes', 'no'], markerStart=0.5, textSize = 2.0, showAccept=False, singleClick=True)
        ratingScale1 = visual.TextStim(win, text = 'YES', pos = (-0.25,-0.25), height=0.1, wrapWidth=None, ori=0, depth=0.0)
        ratingScale2 = visual.TextStim(win, text = 'NO', pos = (0.25,-0.25), height=0.1, wrapWidth=None, ori=0, depth=0.0)



		#Prepares the Scale
        ratingScale1.draw()
        ratingScale2.draw()

    elif rating == 2:
		
		#We will later see how to make sure that the switching between the response buttons is accounted for in the response screen.
        #ratingScale3 = visual.RatingScale(win, choices=['no', 'yes'], markerStart=0.5, showAccept=False, singleClick=True)
        ratingScale3 = visual.TextStim(win, text = 'NO', pos = (-0.25,-0.25), height=0.1, wrapWidth=None, ori=0, depth=0.0)
        ratingScale4 = visual.TextStim(win, text = 'YES', pos = (0.25,-0.25), height=0.1, wrapWidth=None, ori=0, depth=0.0)
        ratingScale3.draw()
        ratingScale4.draw()


    question.draw()           #Prepares the Question.
    win.flip()                #Presents Question and Scale

    
	#The next function () will check if a participant made the right call and for that it is important to tell it what scale we represented 0 or 1.
    CheckResponse()
    win.flip()
    
    #Blank between stimuli to sync with TR
    #
    blank500Clock = core.Clock()
    blank1 = visual.TextStim(win=win, name='blank1',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-1.0, ); 
	
	#add duration or length, 3 sec + the array with the [in_counter]
    
    blank1.draw()
    win.flip()
    
    core.wait(3+ jitter_array[in_counter])


	#Now this function does what its name says but want an argument to be send with when called upon. The name is arbitrary but decides what I have to say in this function to get access to the send argument. In our case this is the generated number. The words WelchAuswahl and scale are just placeholders and interchangeable. CheckResponse(scale) does not care what you put into the brackets when you call it but it does care what that stands for because what it stands for, might not work with the code parts that use the transferred argument scale now stands for. Imagine it as a letter from one function to the next. :)
	
def CheckResponse():

    #I am starting a new clock here to make sure the Questionscreen is presented a whole second even if the participant decides right away if he has seen this person beforehand or not.
    timer = core.Clock()
	
	#Wait for button press. We only wait 0.9sec because psychopy doesnt work with timing it works on frame rates, yet we need timings for the MRI scanner's TR (time to repeat) to be aligned with our stimuli presentation.

    keys = event.waitKeys(maxWait=2, keyList = ['left', 'right','escape'], clearEvents = True)

    try:
        logging_list.append(str(clock.getTime()) + str(keys[0])+' button pressed')
        onset_dict.append(str(clock.getTime()))
        event_dict.append(str(keys[0])+' button pressed')
        logging.log(level=28, msg=str(keys)+' key_pressed_by_participant')
    except:
        logging_list.append(str(clock.getTime()) + 'not pressed'+' key_pressed_by_participant')
        logging.log(level=28, msg=str(keys)+' key_pressed_by_participant')
        onset_dict.append(str(clock.getTime()))
        event_dict.append(' button not pressed')
	#placing the pressed button in our logfile

	
	#here we make sure the question screen is presented at least 1sec as described above.
    if timer.getTime()<2:

		#getTime() returns the time since creation of the clock. Since we just created it and ecreate it every iteration its always 0+the time the participant took to answer
        a=2-timer.getTime()
        core.wait(a)
		
	#add pressed key to response list up top	
    responses.append(keys)

    if keys == None:
        pass
    elif keys[-1] == 'escape':
        win.close()
        core.quit()
		
    #change = True is a security check for the next few lines. If we just ask which scale was used and what the last key response was then switching left to right would make the next statement scale being 1 and the last keypress being right also true and switch it back to left.
    change = True

	#responses[-1] refer to the last element in that list, which is our key press atm.
    if rating == 2 and change == True and responses[-1]==['left']:
        responses[-1]=['right']
        change = False             #Setting change=False here stops the elif statement from reversing what we just did. If the button press was not 'left' then the if statement is false and change stays True.
    elif rating == 2 and change == True and responses[-1]== ['right']:
        responses[-1]=['left']
    else:                          #else: pass is not necessary it is just for overview so I too knwo that there is nothing else happening in this loop.
        pass
		
		
	#these two logging statements might not be necessary. They were implemented to make sure the button presses are switched reliably and only if scale transmitts a 1.

    #logging.log(level=28, msg=str(rating)+' 2=switched 1=notswitched')#comment out if necessary
    #logging.log(level=28, msg=str(responses[-1])+' [left click=yes right click=no None=missed]')
    #logging_list.append(str(clock.getTime()) + str(responses[-1])+' [left click=yes right click=no None=missed]')
    #logging_list.append(str(clock.getTime()) + str(responses[-1])+' [left click=yes right click=no None=missed]')
    #onset_dict.append(str(clock.getTime()))
    #event_dict.append(str(responses[-1])+' [left click=yes right click=no None=missed]')

    #here we just create a little feedback for the participant, so they see what got logged. (only really important for the missed trials)
    if responses[-1] == ['left']:
        answer = visual.TextStim(win, text = 'YES', pos = (0,0), bold = True)
        answer.draw()
        win.flip()
        core.wait(0.25)
    elif responses [-1] == ['right']:
        answer = visual.TextStim(win, text = 'NO', pos = (0,0), bold = True)
        answer.draw()
        win.flip()
        core.wait(0.25)
    else:
        answer = visual.TextStim(win, text = 'you missed', pos = (0,0), bold = True)
        answer.draw()
        win.flip()
        core.wait(0.25)
    #This is where we log a 0 for wrong and missed trials and a 1 for correct responses.
    if responses[-1] == None:
        logging.log(level=28, msg='0'+'  missed trial')
        logging_list.append(str(clock.getTime()) + '0'+'  missed trial')
        onset_dict.append(str(clock.getTime()))
        event_dict.append('0'+' missed trial')
		
	#correct responses are referenced via the next if-loop which place the last image (always on responses[-2] into a dictionary if it is not in there yet. This has to happen after the response check of course to avoid the program thinking that new pictures have been seen before.	
    elif responses[-1] == ['left'] and responses[-2] in alreadyshown:

        #logging.log(level=28, msg='1'+' right:_correctly_remembered')
        logging.log(level=28, msg='1'+' [YES]' + ' [correctly remembered]')
        #logging_list.append(str(clock.getTime()) + '1'+' right:_correctly_remembered')
        logging_list.append(str(clock.getTime()) + '1'+' [YES]' + ' [correctly remembered]')
        onset_dict.append(str(clock.getTime()))
        event_dict.append('1'+' [YES]' + ' [correctly remembered]')

    elif responses [-1] == ['right'] and responses[-2] not in alreadyshown:

        #logging.log(level=28, msg='1'+' right:correct_as_new_identified')
        logging.log(level=28, msg='1'+ ' [NO]'+' [correctly identified as new face]')

        #logging_list.append(str(clock.getTime()) + '1'+' right:correct_as_new_identified')
        logging_list.append(str(clock.getTime()) + '1'+ ' [NO]'+' [correctly identified as new face]' )
        onset_dict.append(str(clock.getTime()))
        #event_dict.append('1'+' right:correct_as_new_identified')
        event_dict.append('1'+' [NO]'+ ' [correctly identified as new face]')

    elif responses [-1] == ['right'] and responses[-2] in alreadyshown:
        #logging.log(level=28, msg='0'+' wrong:_face_not_remembered')
        logging.log(level=28, msg='0'+ ' [NO]'+' [face not remebered]')

        #logging_list.append(str(clock.getTime()) + '0'+' wrong:_face_not_remembered')
        logging_list.append(str(clock.getTime()) + '0'+ ' [NO]'+'[face not remebered]')
        onset_dict.append(str(clock.getTime()))
        #event_dict.append('0'+' wrong:_face_not_remembered')
        event_dict.append('0'+' [NO]'+ '[face not remembered]')

    else:
        logging.log(level=28, msg='0'+ ' [YES]' + ' [WRONG:falsely remembered- FNSB]')#FNSB stands for Face not shown before during the experimental session 
        logging_list.append(str(clock.getTime()) + '0'+ ' [YES]' + ' [WRONG:falsely remembered- FNSB]' )
        onset_dict.append(str(clock.getTime()))
        event_dict.append('0'+ ' [YES]' + '[WRONG:falsely remembered- FNSB]' )
	#after verifying the response the image presented is added to the reference dictionary as explained above.
    if responses[-2] not in alreadyshown.keys():
        alreadyshown[responses[-2]]=1
    else:
        alreadyshown[responses[-2]] += 1 
		#This happens over and over until there are no images left to present so we can go ahead skip back to our first block, which should take us to the endscreen.


#The poolcreator() takes each number we previously got from our Series (crossreferenced from Reihenfolge=[], you remember?) and puts the image of the corresponding identity from a random viewpoint into our facepool=[].
def poolcreator(a):
    #Each identity is supposed to have a maximum of 4 representations on each viewpoint. So we gonna count.
    if a not in frontcounter:
	
	#These counters have corresponding dictionaries={} up top. They are different from lists because here every Key has a corresponding value. The key in our case is the picture and the value is how often it has come along.
        frontcounter[a] = 0
		
	#Initially we create a new entry if we find a key('picture') that is not yet in our counter and since we need 4 of all viewpoints we only check in one dictionary={} and then create the new entry for all three. 	
        leftcounter[a] = 0
		
	#List.append('item') (which we used earlier to put stuff in lists) does not work here because dictionairies={} are fundamentally different from lists=[] and want to be adressed differently. :D	
        rightcounter[a] = 0

    k = frontcounter[a]          #K, l, and m are like reference points for our dictionaries={} so we do not always have to write the whole thing when we want to check what value(count) a key(picture) has.
    l = leftcounter[a]           #This isn't necessary if you replace the letters with their respective dictionairy={} name.
    m = rightcounter[a]

    #ForLorR is an abbreviation for Front or Left or Right. :) randint() is an imported function from the imported library random and random.randint() creates a random number in a given range, here between 0 and 2.
    ForLorR = random.randint(0,2)
	
	
                    #In the following the random number we created will determine the viewpoint of the identity that is put into the facepool.
					
	#If ForLorR is exactly (==) 0 important distinction in python. Equals (=) associat content while exactly (==) test if they mean the same.				
    if ForLorR == 0 and k < 4:
	
	    #And we still have less than 4 faces in our dictionairy={} for the frontview of this specific face. Then first of all it is now one more, so if there is already 1 picture of this view in facepool=[] then there will be 2 in a second and we need to make a note of this before we forget it.
        frontcounter[a] += 1
		
		#Matrix is a matrix that holds all three lists of our images. This is a fancy unecessary step. This could as well say facesfront.append[a]. But I do a Bayesian Statistics and Maschine Learning Module atm so better get used to programming with matrices early on. :)
        facepool.append(matrix[0][a])
		
	#Now we formulate what happens if we already have for pictures of this viewpoint in our facepool (which we know because we did such a good job counting	
    elif ForLorR == 0 and k == 4:
	
		#The consequence is that we increase ForLorR by one so the first condition of our next if-loop is TRUE.
        ForLorR +=1

    #Here all of the above happens again, the picture will just be drawn from a different list=[] in our matrix. I think it is not even a matrix but really just a nested list and only if you fill this nested list with numbers it becomes a matrix. I am currently unsure. Google it.
    if ForLorR == 1 and l < 4:
        leftcounter[a] += 1
        facepool.append(matrix[1][a])
    elif ForLorR == 1 and l == 4:
        ForLorR +=1

	#Same as above with just one difference, we already passed the previous if-loops and we can't easily return to them so we do what is called a recursion.
    if ForLorR == 2 and m < 4:
	
		#A recursion is a function() that calls upon itself. Now this can get you into a whole lot of truble if the boundaries aren't set correctly.
        rightcounter[a] += 1

		#In our case it is ok because Reihenfolge=[] does not contain any number more than exactly 12 times. So when the recursion rerolls the die it will always find a free spot eventually. If a place is found this meansthe other if-loops are no longer True and we do not get lost in an endless loop. :)
        facepool.append(matrix[2][a])
		
	#If this does not immediately make sense have a look at the perks and pitfalls of recursion and iteration online. They both are used to solve similar problems but in a different fashion. 	
    elif ForLorR == 2 and m == 4:
	
	    #Ok, lets go back to StartExperiment() and see what function is called next. (PrticipantInfo())
        poolcreator(a)


#I see you made it. :)
def StartExperiment():

	#The first thing the coder does now is call on the function() AddStimuli, after that he will return here.
    AddStimuli()

	#Here we iterate over every item in our very important list = [].
    for x in range(len(order)):
	
		#This is a tad more complex. S is a Series created with Pandas, which crossreferences every number[x] from Reihenfolge=[] to a randomly generated new number that corresponds to an identity. This process makes sure that Reihenfolge=[] always stays the same but the associated identity is a different one for every participant. :)
        a = faceindex[order[x]-1]
		
		#Now that every x has a new identity they are thrown into our poolcreator() function() and everytime the same x enters S it comes out as the same a assuring what I just said, that the Reihenfolge=[] is never changed just the associated stimuli.
        poolcreator(a)


    #This function() asks for the participant ID.
    ParticipantInfo()
	
	
                            #Next stop AddStimuli().

#The stage is ready, all Stimuli are in their designated places, time for us to inform the participant what he has to do.
def Welcomescreen():
    wlcomScreen1 = visual.TextStim(win, text = 'Welcome to the experimental session. As a reminder, the experiment consist on a facial identity recognition task. In few moments you will see images of faces from different people (BEWARE the following images are different from the training session). All faces will be presented for 750ms. Press either the button underneath your right index or middle finger for further instructions...', pos = (0,0))
	
	#The line above creates a textstimulus in our window. There are loads of thing you can do with that as well. In this line we literally draw() on the screen.
    wlcomScreen1.draw()
	
	#Imagine the above function() as a person drawing something on a screen and this function() as the person turning the screen around so you can see what they did.
    win.flip()
	
	#Here we wait for the participant to indicate that they have finished reading.
    keys = event.waitKeys(keyList = ['left', 'right'], clearEvents = True)
    
    wlcomScreen2 = visual.TextStim(win, text = 'Immidiately after, you will be asked whether or not you remember the face showed to you from previous trials. After each question there will be a fixation screen, during this time try to look at the cross in the middle. Please try to remain still during the session. Press either the button underneath your right index or middle finger for further instructions...', pos =(0,0))
    wlcomScreen2.draw()
    win.flip()
    keys = event.waitKeys(keyList = ['left', 'right'], clearEvents = True)
    
    
    instructScreen1 = visual.TextStim(win, text = 'As we did during the training session, Whenever you are asked whether or not you remember a face please indicate your answer by pressing "left" or "right" button by using your index finger for the left-side button and your middle finger for the right-side button.You will see the question for 2 seconds.Press either the button underneath your right index or middle finger for further instructions...', pos = (0,0))


	#This is the same as the above just a new message. :)
    instructScreen1.draw()
    win.flip()
    keys = event.waitKeys(keyList = ['left', 'right'], clearEvents = True)
    
    instructScreen2 = visual.TextStim(win, text = 'During these 2 seconds you will have to select one of the possible answers. Once you have selected an answer, short feeback will be provide. Beware that the position of the answers might differ from the training session as well. Whenever you are ready please press the button by your index finger, the experiment will start shortly after you pressed this button', pos = (0,0))
    instructScreen2.draw()
    win.flip()
    keys = event.waitKeys(keyList = ['left'], clearEvents = True)
    
    
    #Blank
    blank = visual.TextStim(win=win, name='blank2',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-1.0);

    blank.draw()
    win.flip()


							
							#Next stop: PresentStimuli(), maybe have a look at the first block to see how this new function is part of an iteration over our facepool.

#Yay its done.
def EndExperiment():
    # save everything
    print(logging_list)


    log_dict['onset'] = onset_dict
    log_dict['event'] = event_dict
    f = open('logfile_'+log_dict['subjectID']+'.pkl','wb')
    pickle.dump(log_dict,f)
    f.close()
    #print(log_dict)

    endscreen = visual.TextStim(win, text = 'Thanks for participating', pos = (0,0))
    endscreen.draw()
    win.flip()
    core.wait(2)
    win.close()             #Closes the window
    core.quit()             #Quits the program, they are important for proper termination.


'''
---------------------------------
Start reading here. :)
I tried to make the comments most accessible when first reading through a Block
before scrolling around to look at what the next called function does. ^.^ But you can do whatever you want ofcourse. :D
Just in case you are new to python and/or psychopy, the coder reads from top to bottom.
This means 'def SomeFunctionName()' tells the coder what his abilities are, only when you call a function 'SomeFunctionName()' will python actually do what it says under 'def...()'.
This is why you have to define a function before you can use it. Some functions are included in the imported libraries up top.
---------------------------------
'''
                        #the experiment starts here by calling functions() one by one

#this function() will load the stimuli into their respective list=[] and also ask for the participant ID						
StartExperiment()

#Create a window to draw in, there are loads of things you can change here, have a look at the psychopy documentation. Also this is the cornerstone of stimulus presentation in psychopy.
win = visual.Window(size=(3200, 1800), color='grey', fullscr=True)

#just a clock, you wont ever find more info in anybody else's script xD It is an internal function so no use to look for more details further up top.
clock = core.Clock()

#state where the outputfile should land, the level is important, have a look at the psychopy documentation for logging, there are 4 types of level, 28 allows me to filter what I want to write (fyi 30 is warning messages and 20 to 25 is images and keypresses, a lower level will fill your logfile automatically)
Outputfile = logging.LogFile(f= '\\your\\output\\folder\\for\\output\\log_file\\'+str(responses[0])+'.csv', level = 28, filemode= 'w')


#sets the clock as default for the logfile, good for RT measurement :)
logging.setDefaultClock(clock)

#Instructions and experiment info are presented here
Welcomescreen()
#Trigger test
counter_trigger = 0
while counter_trigger < 5:
    theseKeys = event.waitKeys(keyList=['t'])
    if 't' in theseKeys:
       counter_trigger = counter_trigger + 1

logging_list.append(str(clock.getTime()) +'  ' + str(rating)+' [2=switched 1=notswitched]')
log_dict['answers order']=str(rating)+' [2=switched 1=notswitched]'
# probably wait here for the trigger in the mri scanner

# reset the clock
clock.reset()

jitter_counter = 0
for face in facepool:

	#Transform every face into a visual stimulus. Really important. Like the textstimuli, only that we iterate over our facepool=[] and make them all images that we then put into a new list=[], within which there are the same images in the same order but as vsual stimuli.
    img = visual.ImageStim(win, image = face, size = (0.8, 1.3), pos = (0,0), name = face[-8:-4])
	
	#actually does what it says and is also the main driver of the experiment
    #PresentStimuli(img)
    PresentStimuli(img,jitter_counter)
    jitter_couter = jitter_counter + 1
    #add counter to function call , PresentStimuli(img, counter)
    #add one to the counter counter= counter + 1
    #go to present stimuli function


#Shows the endscreen and also closes the experiment window
EndExperiment()

						
						#Now best continue with StartExperiment(), if you run into a dead end, just return to this part and look for the function that is called next.
						
						
						
						