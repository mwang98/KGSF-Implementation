import os
import json
import pprint


def main():
    context_path = '../data/test_data.jsonl'
    inference_path = 'output_test.txt'
    conversation_path = 'conversation_test.txt'

    conversations = []
    movie_dict = {}

    with open(context_path, 'r') as f_con:
        contexts = f_con.readlines()
        context_raw_data = [json.loads(context) for context in contexts]

    # transform raw data into useful data
    context_data = []
    for context in context_raw_data:
        speakerId = None
        logs = []
        messages_session = []

        for msg in context['messages']:
            if speakerId != msg['senderWorkerId']:
                logs.append(('c', messages_session) if speakerId ==
                            context['initiatorWorkerId'] else ('r', messages_session))

                speakerId = msg['senderWorkerId']
                messages_session = [msg['text']]
            else:
                messages_session.append(msg['text'])

        logs.append(('c', messages_session) if speakerId ==
                    context['initiatorWorkerId'] else ('r', messages_session))

        context_data.append({
            'client': context['initiatorWorkerId'],
            'respondent': context['respondentWorkerId'],
            'logs': logs[1:],
            'conversationId': context['conversationId']
        })

        for key, val in context['movieMentions'].items():
            if not key in movie_dict:
                movie_dict[key] = val

    with open(inference_path, 'r') as f_inf:
        inference = [sen.strip() for sen in f_inf.readlines()]
        inf_idx = 0

        for context in context_data:
            conversation = []
            clientId = context['client']
            responsdentId = context['respondent']

            for idx, utter in enumerate(context['logs']):
                if utter[0] == 'c':
                    conversation.append({'c': utter[1]})
                else:
                    assert inf_idx < len(
                        inference), f"ConversationId={context['conversationId']}, Inf_idx={inf_idx}"
                    conversation.append({'r': utter[1]})

                    if idx != 0:
                        conversation.append({'i': [inference[inf_idx]]})
                        inf_idx += 1

            conversations.append(conversation)

    # replace movie index by names
    for conversation in conversations:
        for sessions in conversation:
            for key in list(sessions.keys()):
                session = sessions[key]
                idx_at = None
                for i in range(len(session)):
                    while session[i].find('@') != -1:
                        idx_at = session[i].find('@')
                        if session[i][idx_at:].find(' ') != -1:
                            idx_end = session[i][idx_at:].find(' ') + idx_at
                        elif session[i][idx_at:].find('.') != -1:
                            idx_end = session[i][idx_at:].find('.') + idx_at
                        elif session[i][idx_at:].find('?') != -1:
                            idx_end = session[i][idx_at:].find('?') + idx_at
                        else:
                            idx_end = len(session[i])
                        movie_id = session[i][idx_at+1:idx_end]
                        if movie_id in movie_dict:
                            session[i] = session[i][:idx_at] + \
                                f"=={movie_dict[movie_id]}==" + session[i][idx_end:]
                        else:
                            session[i] = session[i][:idx_at] + \
                                f"=={movie_id}==" + session[i][idx_end:]

                if key == 'i' and idx_at is not None:
                    sessions['i*'] = session
                    del sessions[key]

    # write to conversation_path
    with open(conversation_path, 'w') as f_out:
        pp = pprint.PrettyPrinter(indent=4, width=200, stream=f_out)
        pp.pprint(conversations)


if __name__ == '__main__':
    main()
